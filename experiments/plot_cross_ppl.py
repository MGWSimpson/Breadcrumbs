
import matplotlib.pyplot as plt

def plot_two_score_lists(scores1, scores2):
    x = list(range(len(scores1)))  # X-axis is the index/position
    plt.plot(x, scores1, label="Scores 1", color='blue')
    plt.plot(x, scores2, label="Scores 2", color='red')

    plt.title("Two Score Lists")
    plt.xlabel("Index")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__": 
    binos_score = []
    ours_score = []
