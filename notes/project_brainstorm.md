Experiment notes:
OLMo 3.1 32B Think

Qwen3-235B-A22B-Thinking

Goodfire paper: models reward-hack on 50–96% of evaluated rollouts


What are the good benchmarks given model capabilities? (some swebench tasks too hard)
P(reward hack)?

*Start by seeing if the models reward-hack



Rough ideas:
We’ll reproduce some of the paper’s results on 2–3 open source models. They’ll have to be smaller than the paper’s; the paper used 1T+ models, and we’ll use models in the 30–180B range probably. One of them will be OLMo so that we can access post-training checkpoints. Specifically, we’ll reproduce:
Their synthetic datasets to build DoM vectors (we’ll build these using the descriptions they post + the composition from the appendix)
Their logit-lens readouts on these DoM vectors, as well as their honeypot steering results.
Detection rates of probes vs monitors on ImpossibleBench and some subset of SWEBench
Extensions:
Put the reward-hacking direction through the J-lens.
Does clamping the reward-hacking direction fry the model’s capabilities?
Steer against the reward-hacking direction.
We want to decompose their reward seeking direction. We’d use DoM to build eval awareness, general cheating, test-related vocabulary (e.g. “secret unit test”), and unrelated baseline contrasts. We’d check the cos similarity of each of these with the reward hacking vector, and regress reward hacking on all of these jointly then take R^2.
We could also project the top candidates out (i.e., the ones with the highest cos or individual R^2) and see how much behavior changes.
Goal: see what makes up this vector; and see if they’re just picking up something spurious (we suspect this may be the case because of how bad their generalization results are, even though they present these results as being good (?))
Could decompose along the verbalization axis
We can recompute the reward hacking vector at a number of OLMo checkpoints and take pairwise cos to see what part of RL instills this vector.
We can also rerun probe tests on these checkpoints.
Goal: see whether reward hacking comes from pretraining, SFT, RLAIF, etc. This is important for figuring out what part of training we need to ’fix.’
We want to RL one of the models with an explicit penalty from the probe. So we can train it on a mixture of possible and impossible tasks, but on each rollout run the probe and include a penalty for it firing (probably a continuous penalty). We’d also track the reward hack rate.
Thus, we can see if the model learns to route reward hacking around the probe. If so, it might be a very bad idea to use this probe during training as the paper suggests!
Penalty, clamping, discarding samples
Compare to rubric judge (RL with LLM as a judge reading everything, and penalize rollouts that are bad)
Notes:
When removing/isolating the relevant components, retest on various tasks to see if we’re actually removing “hacking” vs. just making the model worse
Comments from Lia: This seems ambitious for a two week project. The reward hacking decomp sounds v interesting, and the RL can probably be a stretch goal if you finish the first two. Also would be careful with removing/isolating the components, I'd retest on the tasks to see if you're actually removing "hacking" vs just making the model worse
Additions:
The emotion vectors paper found that steering with the ‘desperation’ vector raises reward hacking from roughly five percent to roughly seventy percent across a narrow range of steering strengths. 
It might be interesting to regress the reward hacking direction onto emotion space, enforcing some kind of sparsity constraint. 
Idea for how this might be implemented: emotions tab


Assignment details:
Deliverable
Short report (3-5 pages) + github
Recreated headline result (figure or table).
1–2 extensions (divergences).
Reflection on robustness / fragility of findings.
Code + README.
Presentation up to 8 min long
Due date: 10/18
