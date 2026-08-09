# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

# Ideas so far
## Reward Shaping
Traditional simple RL problems can be run so that the agent attempts to reach the direction with each step resulting in
a -1 reward while getting to the object results in a 1 reward. The events can be used to designate rewards and attach numbers
to each of the events stated. Mainly the box exploded, collected coin, killed opponent giving high rewards but the killed
self event giving like -500 reward. 