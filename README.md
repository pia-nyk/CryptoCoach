# Crypto Coach

Deep Q-Learning based Portfolio Optimizer on cryptocurrency prices using DQN Agent. 

<b> Why not use Markowitz frontier directly? </b> <br/>
Markowitz efficient frontier doesn't work in this case as the price returns don't follow Normal distribution, which is a crucial assumption in the Modern portfolio theory. This is practically verified. [Pls refer this jupyter notebook](Portfolio_optimization_efficient_frontier.ipynb)

<b> What does the agent do? </b> <br/>
The DQN agent is trained to predict the prices & hence the returns obtained on 3 actions - (selling, buying and holding) the crypto portfolio consisting of 15 stocks by assigning them appropriate weights dynamically.

<b> How is the efficiency of the agent's actions tested? </b> <br/>
The cumulative sum of returns generated by the weights assigned dynamically by the agent is compared with a baseline (returns in case equal weightage is given to all the stocks in the portfolio throughout). 

<b> Results </b> <br/>
<img src="https://github.com/pia-nyk/Deep-Q-Learning-based-portfolio-optimizer/blob/master/reinforcement_learning/results.png" width="500" height="500">

<b> References </b> <br/>
<ul>
<li>This project is inspired by the medium article. The original article uses a single deep NN, this has been modified to use 2 Deep NNs - TrainNet & TargetNet: https://medium.com/swlh/ai-for-portfolio-management-from-markowitz-to-reinforcement-learning-cffedcbba566</li>
<li> To have a better intuition of DQN code (since it's fragmented across files in this case), pls refer: https://github.com/pia-nyk/RL-on-OpenAI-Gym/blob/master/DQN/cart-pole.py
<li>If you wish to learn more about Portfolio optimization, pls check: https://github.com/pia-nyk/Portfolio-Construction-and-Analysis-Using-Python </li>
</ul>
