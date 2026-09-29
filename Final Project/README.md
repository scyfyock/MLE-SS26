# Report Abstract
In this paper, we compare the performance of Q-learning
models that utilize world models for training and planning
on the task of winning and scoring points in Bomberman.
We built several agents for this task with varying world
models, and one without. The first model we made is DynaQ,
a tabular Dyna-Q whose world model replays previously
observed transitions for background planning. The second
model is QWM, a deep Q-learning agent with a learned World
model built on existing architecture. QWM uses its learned
world model for decision-time planning through a tree search.
Using an average of 1000 rounds against rule based agents
as a benchmark for evaluating our models, QWM achieved
an average score/round of 3.87, survived 63% of the rounds
and with only 242 suicides. Comparatively, DynaQ achieved
a mean score of 3.71, survived 49.8% of the rounds, but had
408 suicides. The closeness in score of the models required
further testing against each other, so various head-to-head
scenarios were run against the two. The mean scores continued
to not be statistically significant, but the QWM agent showed
a clear victory in survival and a low number of suicides. We
conclude that the architecture of QWM is our best model and
an apt architecture for the given problem of gaining points and
winning at Bomberman, though its advantage lies in survival,
and comes at the cost of a much longer inference time.

# Bomberman RL – QWM Agent

The QWM agent is located under the training_for_qwm folder within the Final Project
folder.

# Bomberman RL – DynaQ Agent

The DynaQ agent is located under the dyna_agent folder within the 
Final Project/agent_code folder. The different schema versions are labeled 
under dyna_agent_schema3 and dyna_agent_schema7.