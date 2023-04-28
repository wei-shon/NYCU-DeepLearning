import pandas as pd
import matplotlib.pyplot as plt
file = open('output.txt' , 'r')

episodes = []
mean_score = []
score = 0
timestep = 0 
episode = 0
for line in file:
    score += int(line.splitlines()[0])
    timestep+=1
    episode +=1
    if timestep == 1000:
        mean_score.append(score/1000)
        episodes.append(episode)
        score=0
        timestep=0
    # print(line)

plt.plot(episodes, mean_score)
plt.title("TD-2048") # title
plt.ylabel("Score") # y label
plt.xlabel("Episode") # x label
# print(len(mean_score))
plt.show()