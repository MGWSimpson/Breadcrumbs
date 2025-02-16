import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Опционально используйте те же утилиты из "utils" или "metrics", 
# если они есть в вашем проекте
from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)  # если нужно для приватных моделей
}

TRINOCULARS_DEFAULT_THRESHOLD = 0.90  # Примерный порог, подбирается вручную или валидируется

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
DEVICE_3 = "cuda:2" if torch.cuda.device_count() > 2 else DEVICE_1


class Trinoculars(object):
    """
    Пример класса, который использует три модели (M1, M2, M3)
    и агрегирует их "Binoculars Scores" в одну финальную метрику.
    """

    def __init__(
        self,
        model1_name_or_path: str,
        model2_name_or_path: str,
        model3_name_or_path: str,
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        threshold: float = TRINOCULARS_DEFAULT_THRESHOLD,
    ) -> None:
        """
        Параметры:
        ----------
        model1_name_or_path, model2_name_or_path, model3_name_or_path : 
            пути/названия моделей для AutoModelForCausalLM.
        use_bfloat16 : bool
            использовать ли bfloat16 (True) или float32 (False).
        max_token_observed : int
            максимальное число токенов при tokenize (усечение).
        threshold : float
            порог для бинарной классификации (машинный/человеческий).
        """
        # Проверим, что у всех одинаковый токенизатор (если это критично для cross-PPL)
        assert_tokenizer_consistency(model1_name_or_path, model2_name_or_path)
        assert_tokenizer_consistency(model2_name_or_path, model3_name_or_path)

        self.threshold = threshold

        # Загружаем модели
        self.model1 = AutoModelForCausalLM.from_pretrained(
            model1_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )
        self.model2 = AutoModelForCausalLM.from_pretrained(
            model2_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )
        self.model3 = AutoModelForCausalLM.from_pretrained(
            model3_name_or_path,
            device_map={"": DEVICE_3},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        # Токенизатор (общий) - возьмём от model1, 
        # но предварительно проверили assert_tokenizer_consistency
        self.tokenizer = AutoTokenizer.from_pretrained(model1_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

    def free_memory(self) -> None:
        """Пример высвобождения памяти, если надо выгружать модели."""
        devices = [self.model1, self.model2, self.model3]
        for m in devices:
            m = m.to('cpu')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        del self.model1
        del self.model2
        del self.model3
        self.model1, self.model2, self.model3 = None, None, None

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        """Подготовка входных данных (BatchEncoding) для моделей."""
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        )
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding):
        """
        Прогоняем batch через все три модели.
        Возвращаем три тензора logits1, logits2, logits3
        """
        # Переносим batch на нужные устройства
        encodings_1 = encodings.to(DEVICE_1)
        encodings_2 = encodings.to(DEVICE_2)
        encodings_3 = encodings.to(DEVICE_3)

        logits1 = self.model1(**encodings_1).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize(device=DEVICE_1)

        logits2 = self.model2(**encodings_2).logits
        if DEVICE_2 != "cpu":
            torch.cuda.synchronize(device=DEVICE_2)

        logits3 = self.model3(**encodings_3).logits
        if DEVICE_3 != "cpu":
            torch.cuda.synchronize(device=DEVICE_3)

        return logits1, logits2, logits3

    def compute_score(self, input_text):
        """
        Возвращаем "финальный" TriNoculars Score для одного или нескольких текстов.

        Возможные реализации:
         - Усреднение трёх Binoculars Scores (B_{12}, B_{13}, B_{23})
         - Или любая другая агрегация (максимум, минимум, медиана и т.д.)
        """
        is_string_input = isinstance(input_text, str)
        batch = [input_text] if is_string_input else input_text

        # Токенизируем общий batch
        encodings = self._tokenize(batch)
        # Получаем logits со всех трёх моделей
        logits1, logits2, logits3 = self._get_logits(encodings)

        # Раскладываем всё на одно устройство (например, DEVICE_1) для удобства 
        # при вычислении perplexity / cross-entropy
        encodings_ = encodings.to(DEVICE_1)
        logits1_ = logits1.to(DEVICE_1)
        logits2_ = logits2.to(DEVICE_1)
        logits3_ = logits3.to(DEVICE_1)
        pad_id = self.tokenizer.pad_token_id

        # Вычисляем PPL для M1 и M2 
        # (можно по желанию и для M3, если хотим ещё одну комбинацию)
        ppl1 = perplexity(encodings_, logits1_)
        ppl2 = perplexity(encodings_, logits2_)

        # Cross-perplexity для пары (M1, M2), (M1, M3), (M2, M3)
        x_ppl12 = entropy(logits1_, logits2_, encodings_, pad_id)
        x_ppl13 = entropy(logits1_, logits3_, encodings_, pad_id)
        x_ppl23 = entropy(logits2_, logits3_, encodings_, pad_id)

        # Пример: рассчитываем три Binoculars Scores:
        #   B12 = PPL(M1)/X-PPL(M1,M2)
        #   B13 = PPL(M1)/X-PPL(M1,M3)
        #   B23 = PPL(M2)/X-PPL(M2,M3)
        B12 = ppl1 / x_ppl12
        B13 = ppl1 / x_ppl13

        # для B23 берём ppl2 / x_ppl23
        B23 = ppl2 / x_ppl23

        # Агрегация: возьмём просто среднее (можете менять логику)
        final_score = 0.3333 * (B12 + B13 + B23)

        # Преобразуем в list/float
        final_score = final_score.tolist()

        if is_string_input:
            return final_score[0]  # Для одного текста возвращаем scalar
        else:
            return final_score  # Для batch возвращаем список

    def predict(self, input_text):
        """
        Бинарная классификация с использованием мажоритарного голосования:
        Если 2 из 3 итоговых Binoculars Scores (B12, B13, B23) меньше порога,
        то возвращается "Most likely AI-generated", иначе – "Most likely human-generated".
        """
        is_string_input = isinstance(input_text, str)
        batch = [input_text] if is_string_input else input_text

        encodings = self._tokenize(batch)
        logits1, logits2, logits3 = self._get_logits(encodings)

        encodings_ = encodings.to(DEVICE_1)
        logits1_ = logits1.to(DEVICE_1)
        logits2_ = logits2.to(DEVICE_1)
        logits3_ = logits3.to(DEVICE_1)
        pad_id = self.tokenizer.pad_token_id

        ppl1 = perplexity(encodings_, logits1_)
        ppl2 = perplexity(encodings_, logits2_)

        x_ppl12 = entropy(logits1_, logits2_, encodings_, pad_id)
        x_ppl13 = entropy(logits1_, logits3_, encodings_, pad_id)
        x_ppl23 = entropy(logits2_, logits3_, encodings_, pad_id)

        B12 = ppl1 / x_ppl12
        B13 = ppl1 / x_ppl13
        B23 = ppl2 / x_ppl23

        # Если на вход одно предложение, то вычисляем голосование для одного экземпляра
        if is_string_input:
            vote_count = (B12 < self.threshold) + (B13 < self.threshold) + (B23 < self.threshold)
            return "Most likely AI-generated" if vote_count >= 2 else "Most likely human-generated"
        else:
            predictions = []
            # Если результаты представлены как тензоры, приводим к numpy-массивам
            if hasattr(B12, "cpu"):
                B12 = B12.cpu().numpy()
                B13 = B13.cpu().numpy()
                B23 = B23.cpu().numpy()
            for b12, b13, b23 in zip(B12, B13, B23):
                vote_count = (b12 < self.threshold) + (b13 < self.threshold) + (b23 < self.threshold)
                predictions.append("Most likely AI-generated" if vote_count >= 2 else "Most likely human-generated")
            return predictions
