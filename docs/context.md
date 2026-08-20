
Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Research
August 19, 2026
Generalist Team
17 min read
Table of Contents
Introduction
Introducing GEN-1.5
Scaling Pretraining for Robotics
One-Shot Learning In-Context
Compositional Generalization
Zero-Shot Sim-to-Real Transfer
Human-to-Robot In-Context Learning
Few Gradient Step Adaptation
Physical Generalization
Looking Ahead
Citation
Back to Blog
GEN-1.5Embodied Foundation Models are One-Shot Learners
Humans have a remarkable ability to perform new physical skills from only one or a few examples. Our latest robot foundation model, GEN-1.5, exhibits the beginnings of that same ability: it can learn a new task in seconds, from a single example, without gradient updates or fine-tuning. It displays broad capabilities across one-shot and few-shot learning from demonstration, as well as zero-shot physical generalization. Although the tasks are simple and short-horizon, this is the first model we know for which one-shot and few-shot learning of physical skills have emerged at scale. We view these results as a significant step towards our mission of building general intelligence for the physical world.


The promise of a robot foundation model is simple to state: walk up to a robot, and get it to do any task almost immediately. Whether zero-shot, one-shot, or few-shot, what matters from a capability standpoint is immediacy and generality: can the model learn a new task quickly and generalize to new situations? For physical tasks, this level of intelligence demands both broad abilities in comprehending task intent, as well as adapting in real time, closed-loop, to the unexpected variation of the real world.

For language models, the ability to quickly learn new tasks from just one or a few examples arrived as a hallmark capability of GPT-3.1 Across a broad suite of language tasks, it achieved roughly 45% average accuracy with one-shot in-context prompting without training, and up to ~65% with few-shot (~100 examples).1 Models prior to GPT-3 had shown flashes of zero-shot ability and even initial few-shot results2, but GPT-3 achieved significantly broader few-shot performance, paired with what at the time was an immense step in generalization capabilities.

In robotics, the analogous pursuit of systems that could generalize a task from one or a few demonstrations, has persisted for decades — tracing back at least to the teach-by-guiding of the 1954 Unimate patent3 and MIT’s 1970 Copy Demo.4 A large number of prior works, including our own,5 have shown various forms of in-context learning but over a limited set of task variations, or under restrictions to particular objects, task types, or sensing modalities.6,7,8,9,10,11 The ability to learn closed-loop physical skills from just one or a few demonstrations, and to do so across a broad range of tasks without such restrictions, has predominantly been considered out of reach. Such an ability may also likely be underpinned by a foundation that enables other broad generalization capabilities.

Language model one-shot example
Q: Who wrote Romeo and Juliet?
A: William Shakespeare

Q: Who wrote War and Peace?
A: Leo Tolstoy

Embodied model one-shot example
Marker Into Cup
Pour Bolts
Zipper

Play Video

0:00

0:23


One-Shot Example: Marker Into Cup
Figure 1. One-shot learning in-context with language models and embodied models. In the language model example, the output of the model is highlighted in green. In our example, the prompt is a sensorimotor sequence from the human demonstration data, and GEN-1.5 controls the robot to accomplish the inferred task.
Introducing GEN-1.5
We’ve created GEN-1.5, our latest robot foundation model that exhibits broad one-shot and few-shot learning from demonstration capabilities, as well as zero-shot generalization, e.g. improvisation and novel tool use (e.g. brush, dustpan, etc.). GEN-1.5 is a large multimodal model that processes video input (30 seconds of memory, alongside other sensor, language, and proprioceptive inputs) and produces 100 Hz action trajectories. Its capabilities include:

One-shot learning via in-context prompting. The model learns new tasks in seconds when prompted with 3 to 12 seconds of a single demonstration, no training required. We refer to the use of sensorimotor examples in the context window as “physical prompting.”
Compositional generalization. Given two different physical prompts in context, the model chains them into a single longer-horizon behavior.
Zero-shot sim-to-real transfer. A demonstration recorded in simulation works as a physical prompt for a real-world task, even though pretraining contains no simulation data.
Human-to-robot imitation. In some cases a person can demonstrate a task with their own hands, in view of the robot’s cameras, and the model reproduces it with the robot’s hands.
Few-shot adaptation via gradient descent. The model can be fine-tuned to a new task in 1–10 gradient steps on 1–5 minutes of data (~10–50 demonstrations).
Improvising new strategies and tool use. The model generalizes at the level of behavioral strategies: forming entirely new trajectories to reach a goal, using unseen tools (e.g. brush, dustpan, etc.) to create new solutions to tasks demonstrated with other tools, and working ambidextrously even when prompted or fine-tuned to perform the task with a specific hand.
These capabilities appear to emerge directly from pretraining on large amounts of physical interaction data. We did not explicitly train for any of them: no architectural changes to promote in-context learning, no inner or outer meta-learning loop12 pressuring the model to adapt from minimal data, no auxiliary objectives13 encouraging improvisation. To our surprise, GEN-1.5 does this across a broad range of physical tasks out of the box.

Twist lid off glass jar
Unzip pencil pouch
Brush cube into bowl
Remove vacuum pad
Physical prompt

Play Video

0:00

0:04


Physical Prompt: Twist Lid Off Glass Jar
Model rollout

Play Video

0:00

0:03


Model Rollout: Twist Lid Off Glass Jar
Experiments across 10 diverse tasks show 59% (±10% std. dev.) average success with one-shot in-context prompting, straight from the pretrained model. With few-shot learning, performance rises to 83% (±9% std. dev.) via 10 gradient steps on 5 minutes of data per task (~50 demonstrations). In some cases, in-context learning a new task exceeds the performance of 1–5 gradient steps on the same demonstration data. Although the tasks are simple and short-horizon, and the success rates are modest, this is the first model we know of that has demonstrated the general ability to learn a wide range of dexterous closed-loop physical tasks from just one-shot or few-shot demonstrations.

10 gradient steps (5 min of data)
In-context learning (12s of demos)
Retrieve money from purse
83.3%
60.7%
Fold and crease paper
69.3%
50%
Twist lid off glass jar
94.5%
60%
Stack two small cups
75%
67%
Sweep trash with brush
99%
37.3%
Open book cover
82.7%
54.7%
Brush cube into bowl
71.2%
60.8%
Flip phone upside down
81%
78%
Unzip pencil pouch
86%
55.5%
Remove vacuum pad
86%
64%
0%20%40%60%80%100%
Task Success Rate (%)
Figure 2. GEN-1.5 can learn short-horizon atomic manipulation tasks at 83% average success rate after 10 steps of gradient descent on 5 minutes of data per task, and 59% with zero gradient updates and 3 to 12 seconds of demonstration data using emergent in-context learning.
Scaling Pretraining for Robotics
Over the past two years, we’ve been focused on building a pretraining engine for scaling embodied foundation models trained from the ground up on physical experience, alongside algorithmic improvements that have compounded the rate of progress. As we announced nine months ago, leading up to GEN-014 we started to see predictable scaling laws.15 Five months later, we announced GEN-1,16 which demonstrated the ability to be post-trained to task mastery at 99%+ success rates and showed initial signs of improvisational intelligence.

GEN-1.5’s initial pretraining began in parallel — it has now been training continuously for over eight months. We left it running because every metric we tracked kept improving with the engine: absorbing more data, scaling more efficiently with compute, and achieving step-change gains with successive surgical architectural and algorithmic changes. It was clear that the model was getting better, and the trend was consistent: new tasks were becoming more data-efficient, more compute-efficient, and more general.

1.2 × 10−2
1.4 × 10−2
1.6 × 10−2
1.8 × 10−2
2.0 × 10−2
2.2 × 10−2
Dec
2025
Jan
Feb
Mar
Apr
May
Jun
Jul
Aug
2026
Month
Next Action Prediction Error (Validation)
Phase 1
Phase 2
Phase 3
Figure 3. GEN-1.5 has been pretraining on our data engine for over 8 months, and continues to improve its next action prediction error on a held-out validation set across 3 training phases.
As the model continued to train, we began experimenting with how few finetuning steps we could use to adapt to new tasks, finding the model could learn new tasks from 100s, then 10s, then eventually, 1 gradient step on just one minute of data. As far as we know, the ability to learn skills with such few gradient steps had not been observed before. We then asked, can this model learn new tasks without training, purely in-context and with zero gradient steps? That this works at all changes how we think about how these models can be used, about their potential impact, and the road ahead for building general physical intelligence.

One-Shot Learning In-Context
GEN-1.5 can be prompted with a single demonstration inserted into its 30-second context window, and the remainder holds rolling observations. Physical prompts are sensorimotor examples (i.e. sensor data plus action trajectories), recorded either as human data (with a pair of handheld grippers) or as rollouts from the robot itself. Once the prompt is in context, the model performs the task immediately, with no training steps. The performance of one-shot learning in-context is modest (59% average success across diverse tasks including handling zippers, opening jars, grabbing money out of wallets, etc.), but the fact that inserting a single demonstration in the context buffer, without ever training for it, yields any measurable competence at all was unexpected. This drastically accelerates reaching a base level of performance that can be subsequently refined towards mastery.16 Skills learned in-context are currently more brittle than finetuned models, but can generalize to some perturbations, improvise, and recover from mistakes.


Play Video

0:00

0:50


Physical Prompt Engineering Interface
Physical prompt engineering involves a drag and drop interface to select which demonstrations to be inserted into the model’s context window. This live recording shows physical prompting the model to learn two different tasks back-to-back: (i) unzipping a pencil pouch, and (ii) retrieving money from the pouch.
We did not explicitly train GEN-1.5 for in-context learning, and the tasks we tested were not engineered into the pretraining data beforehand. This is a general model which we are prompting without regard to the pretraining data distribution.

Why this capability emerges from pretraining is difficult to pinpoint. One hypothesis, by analogy to language, is that the distribution of physical observations and actions may exhibit “burstiness” and Zipfian structure of the kind that has been linked to in-context learning in language models.17 It is also possible that physical work contains naturally repetitive cycles, and the model may have learned to detect and extend such patterns, as language models do with general sequences.18 The model was pretrained on randomly sampled continuous spans from our data engine (activities captured in homes, warehouses, factories, and elsewhere) with no bespoke infrastructure for packing examples into context — physical prompts introduce discontinuous jumps in time that the model never saw in training.

Robotics is inherently multimodal; and as in human learning, there are many ways to teach a robot something new — the two options of either (a) demonstrations or (b) language instructions are perhaps the most natural for having humans specify tasks.19 While language suffices for some task specifications, many physical actions are difficult to precisely describe in language20 (e.g. it is far easier to show exactly how to seat two Lego bricks than to say it). Prompting a task in native observations and actions is also a more comprehensive test of sensorimotor understanding: the model must infer the goal from the demonstration, repurpose existing knowledge, and improvise under new initial conditions.

Compositional Generalization with Physical Prompt Engineering
Physical prompts can also be composed. For example, if we place demonstrations of two different tasks in context (each recorded independently, with no transition between them), GEN-1.5 can chain them into one continuous behavior, performing one task and then flowing into the next. The model bridges the two on its own, producing intermediate motions (repositioning, regrasping, error recovery) that appear in neither demonstration.

Physical prompt A

Play Video

0:00

0:09


Physical Prompt A
Physical prompt B

Play Video

0:00

0:07


Physical Prompt B
Model rollout of A + B

Play Video

0:00

0:26


Model Rollout of A + B
Composing physical prompts. Two demonstrations of different tasks: (a) unzip pencil pounch, and (b) retrieve money, are placed together in the model’s context. The model chains them into one continuous behavior, bridging the two with intermediate motions, recoveries, and ambidexterity that appear in neither prompt.
In practice, this opens up “physical prompt engineering”: rather than collecting a demonstration of a full compound task, one can assemble it from a small library of short, reusable physical prompts. As models improve, composing skills in context may become a practical way to program longer-horizon behaviors; the physical analogue of chaining instructions in a language prompt.

Zero-Shot Sim-to-Real Transfer with In-Context Learning
In-context learning also crosses the sim-to-real gap. A prompt can be formed entirely from simulated experience (e.g., from a scripted policy, an RL agent, or a human teleoperating a simulated robot) and be used to prompt the real robot. To clarify, “zero-shot sim2real transfer” typically refers to training a policy in a simulator on a particular task, then running that policy in the real world without real-world data for that task. In the case we show here, however, the model was not trained on the task in either the simulator or the real world.

Prompt from simulation

Play Video

0:00

0:03


Prompt From Simulation
Model rollouts in real world

Play Video

0:00

0:27


Model Rollouts in Real World
Zero-shot sim2real transfer. A demonstration recorded entirely in simulation (left) is placed in the model’s context as a physical prompt, and the real robot performs the task (right) — despite zero simulation data in pretraining. The prompted behavior generalizes to different hands and new object positions and sizes.
GEN-1.5 pretraining contains no simulation data, neither rendered video nor simulated dynamics, yet the model can be effectively prompted by rollouts from the simulator. The prompted behaviors then generalize as other physically prompted behaviors do: to different hands, and to new object positions and sizes in the real scene. For a subset of tasks, this means demonstrations no longer need to be collected physically — they can instead be gathered by whichever means inside a simulator.

Human-to-Robot In-Context Learning
In some cases, in-context learning transfers across the embodiment gap entirely: a human demonstrates a task with their own hands, observable through the robot’s cameras, and the robot can reproduce it immediately afterward.


Play Video

0:00

0:30


Human-to-Robot In-Context Learning
Few Gradient Step Adaptation
GEN-1.5 can also adapt to new physical tasks in extremely few gradient steps, as few as 1 to 10. Typically, training previous robot models on a new task can take tens of thousands of gradient steps (sometimes orders of magnitude more), but a hallmark capability of foundation models21 is that they can be rapidly adapted to new tasks with a small amount of fine-tuning. While this mechanism can be built in explicitly e.g. with second order gradients encouraging fast adaptation,12 we find that the pretrained base of GEN-1.5 already adapts in very few steps without any such machinery.

0.1
1
10
θGEN-1.5 pretrain
θflip phone upside down
θfold and crease paper
θopen book cover
θunzip pencil pouch
θremove vacuum pad
θretrieve money from purse
θbrush cube into bowl
θstack two small cups
θsweep trash with brush
θtwist lid off glass jar
Figure 4. Few-step adaptation moves the pretrained weights in a different direction for each task, viewed as a classical MDS embedding with pairwise L2 distances between weights drawn radially logarithmic around the pretrained model.
Importantly, such an extremely small amount of training steps requires orders of magnitude less task-specific compute, and opens up a much more flexible view of task adaptation than heavy finetuning. It may be more apt to describe this as test-time training22 in an extremely low-data regime. Test-time training commonly uses tens of gradient steps; GEN-1.5 learns a new physical task in 1–10 steps on 5 minutes of data. Ten steps change the model weights on held out tasks by less than 0.15%, suggesting that fine-tuning slightly reconfigures knowledge already present rather than building new representations.

In our experiments for 10-step adaptation, we sample sequences from 5 minutes of data and train with gradient descent using hyperparameters similar to pretraining. In the extreme one-step regime, sampling from one minute of data, success on a held-out task is 66.5%, and performance improves with larger batch sizes and higher learning rates. We did not tune this procedure or sweep adaptation-specific hyperparameters; these results come largely out of the box.

Physical Generalization
For every task above, fine-tuned models generalize well beyond their demonstrations — not only to new embodiments, object instances, and environments, but also to fundamentally different manipulation strategies for the same goal: alternative grasps and motions, clearing obstacles, and using tools absent from the fine-tuning data.

Novel tool use improvisation. In one example, we demonstrated using a brush to sweep a block into a bowl, and fine-tuned the model on 5 minutes of human demonstrations. The model was able to figure out how to use a variety of other tool options besides the brush in order to accomplish the task. When presented with a banana, it used the banana as a makeshift brush. When presented with a dustpan, however, the model exhibited a larger strategic departure from its demonstrations, and through a variety of means would use the dustpan to lift up the block and dump it into the bowl. Neither the fine-tuning data nor, to the best of our knowledge, the pretraining data contains a dustpan used this way, and the nearest pretraining examples bear little resemblance to the task. Handed a dustpan, the model composed an entirely new contact sequence to complete the task out of the box, with no language guidance. Below are example rollouts from this model.

Play Video

0:00

0:03


Brushing a Block Into a Bowl (Expected)

Play Video

0:00

0:04


Brushing a Block Into a Bowl (Improvisation)

Play Video

0:00

0:06


Brushing a Block Into a Bowl (Improvisation)
Banana as an impromptu brush. Improvising by using a banana to sweep the block into the bowl.

Play Video

0:00

0:09


Banana, Block, and Bowl Improvisation
Multiple blocks. Brushing multiple blocks into the bowl.

Play Video

0:00

0:23


Brushing Multiple Blocks Into a Bowl
Ambidexterity. Brushing the block with either hand even though the demonstrations only use one.

Play Video

0:00

0:07


Ambidextrous Brushing
Fine-tuning data. Example human demonstration of brushing a block into a bowl (model view).

Play Video

0:00

0:03


Human Demonstration
Most similar tasks from pretraining. These are approximately the closest activities using nearest neighbor language search over 1,891,392 scenes.

Play Video

0:00

0:03


Pretraining Nearest Neighbor 1

Play Video

0:00

0:05


Pretraining Nearest Neighbor 2

Play Video

0:00

0:05


Pretraining Nearest Neighbor 3

Play Video

0:00

0:04


Pretraining Nearest Neighbor 4
The ability to improvise under unexpected situations appears to be central to how the model masters new tasks: it can correct its own mistakes, sometimes before they occur. We first observed traces of this behavior in GEN-1.16 In GEN-1.5 it is both more frequent and more sophisticated, and it strengthens as the number of fine-tuning gradient steps decreases, presumably because lightly adapted models stay closer to their pretrained priors and can draw on a broader repertoire of behaviors when the situation departs from the demonstrations.

Handling obstacles. Although the model was only fine-tuned to put the block into a bowl, it appears to be able to remove obstacles (like a piece of paper covering the bowl) to complete the task, and sometimes place the paper back on top of the bowl. There was no paper covering the bowl in the 5 minutes of task-specific data (with which the model was fine-tuned for only 1 gradient step), and no such task in this setting (to the best of our knowledge) was in the pretraining data.

Play Video

0:00

0:23


Handling Obstacles
Fine-tuning data

Play Video

0:00

0:03


Fine-Tuning Data
Model rollouts

Play Video

0:00

0:17


Model Rollouts
More examples of intriguing emergent improvisation behaviors:

Removing obstructions. When a Lego brick gets unexpectedly stuck on the fingertips, the model uses the other hand to remove them.
Model rollout

Play Video

0:00

0:10


Removing Obstructions
Fine-tuning data

Play Video

0:00

0:05


Fine-Tuning Data
Bimanual coordination. The model sometimes uses two hands to rotate a jar lid (with a fundamentally different contact and motion strategy), when the training data was only using one.
Expected model rollout

Play Video

0:00

0:07


Expected Model Rollout
Improvised model rollout

Play Video

0:00

0:04


Improvised Model Rollout
Tendency to organize. Models fine-tuned only to place a single block into a single bowl sometimes exhibit more general behaviors such as sorting blocks by color or category (a generalized form of physical commonsense20).

Play Video

0:00

0:49


Cube Sorting 1

Play Video

0:00

0:45


Cube Sorting 2
Generalizing to new objects. Models fine-tuned for 10 gradient steps on 5 minutes of data to to twist off a lid jar can generalize to cups and bottles it’s never seen before, which requires reasoning over where to grasp with both hands, and how to uniquely rotate the wrist to twist each lid off.

Play Video

0:00

0:04


New Object Generalization: Jar

Play Video

0:00

0:08


New Object Generalization: Bottle

Play Video

0:00

0:04


New Object Generalization: To-Go Cup

Play Video

0:00

0:06


New Object Generalization: Container
Looking Ahead
When we started this journey, we did not set out to specifically build a one-shot learner. Much of our history as a team has been spent building the fundamental machinery required to iterate on the science of pretraining in robotics from first principles, beginning with a data engine that could fuel the model science with high-quality physical experience at scale.

GEN-1.5 is a milestone we believe to be profound scientifically, not because of higher success rates, but because it represents a new frontier of generality — one that challenges our own understanding of how these models behave when pretrained at a scale of physical interaction data few thought possible without shortcuts. GEN-0 and GEN-1 each gave us increasing confidence that more (and better) pretraining would make adaptation to new tasks more data-efficient. Every trend we measured pointed in the same direction: more pretraining makes adaptation faster, cheaper, and more general. We do not yet see where that curve asymptotes.

What is clear now, and perhaps obvious in hindsight, is that past a certain threshold of pretraining, the cost of adaptation becomes negligible. Emergent in-context learning from a few seconds of data, or one gradient step on one minute of demonstrations, is no longer task-specific training in the conventional sense. It is closer to reminding the model of something it nearly knows, with a tiny amount of compute. That this works at all, changes how we think about how these models can be used, about their potential impact, and the road ahead for building general physical intelligence.

For decades, robots have been marketed as “general-purpose” machines that could in principle do anything — a contrast to the single-purpose factory automation of the past. But that promise was always conditioned on an expert programming them, which took months of effort and specialized knowledge. If interacting with a robot reduces to simply showing it what to do, then two things change fundamentally: how quickly a robot becomes useful (seconds, not months), and who can work with one (anyone).

We are in the early days of our mission to build physical AGI and make it useful to everyone. If you are interested in joining us on this journey, reach out at generalistai.com/careers.

Citation
Please cite this work as

Generalist Team, “GEN-1.5: Embodied Foundation Models are One-Shot Learners”, Generalist AI Blog, Aug 2026.
Or use the BibTeX citation:

@article{generalist2026gen15,
author = {Generalist Team},
title = {GEN-1.5: Embodied Foundation Models are One-Shot Learners},
journal = {Generalist AI Blog},
year = {2026},
note = {https://generalistai.com/blog/gen-1.5},
}
1
Language Models are Few-Shot Learners (Brown et al., 2020)
2
Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
3
Programmed Article Transfer, U.S. Patent 2,988,237 (Devol, filed 1954)
4
The MIT AI Lab Copy Demo (Winston et al., 1970)
5
The Robots Build Now, Too (Generalist, 2025)
6
In-Context Imitation Learning via Next-Token Prediction (Fu et al., 2024)
7
Behavior Prompting Policy: Demonstrations as Prompts for Manipulation (Patel et al., 2026)
8
RoboTTT: Context Scaling for Robot Policies (Jiang et al., 2026)
9
Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Going Beyond World Models & VLAs
Going Beyond World Models & VLAs
Idea
April 7, 2026
Pete Florence
and the Generalist Team
6 min read
Table of Contents
Introduction
Goals are more important than the labels on your tools
How far can we go?
Building for the world that’s coming
Towards physical AGI
Back to Blog
In GEN-1,1 approximately 99% of the parameters are trained from scratch.

Previously, this might be considered wild. For Generalist, it’s a deliberate choice. It follows our strong conviction — pursued for two years — that when you have enough data, you can move faster at pushing the frontier by having complete control over the fundamental model.

GEN-1 is not a fine-tuned vision-language model with robot actions bolted on, nor is it just a world model. It is a first-class-citizen, native foundation model for physical interaction. And there is growing evidence that if you have enough data and compute, training from scratch always wins.2

World models are having their moment in early 2026. VLAs had theirs from 2023 to 2025. Bandwagons are part of the nature of academic research.

At Generalist, we’ve never referred to our models as either VLAs or world models. This is not an accident. We co-invented VLAs,3 have been publishing on world models in robotics4 since 2023, and working on them for a couple years longer than that.

So why no label? For one, your goals are more important than the labels on your tools. And, because you don’t necessarily call a rectangle a square. And, because the supply side will change. We’ll unpack each of these below.

Goals are more important than the labels on your tools
First and foremost, goals are more powerful than methods. John Schulman articulated the comparison well several years ago in a piece5 comparing idea-driven vs. goal-driven research: idea-driven research follows the trends and improves on the latest method, while goal-driven research picks a concrete outcome and solves whatever problems stand in the way. The distinction matters because it shapes what you build and, critically, what you don’t get distracted by. As Schulman argues and I’ve found myself the same, typically goal-driven is the more powerful path.

The current discourse around world models is idea-driven. These are genuinely exciting techniques. But building a world model might not actually be the goal, even for those working on world models. The real question is, what’s your goal?

One example long-term goal we think is worthwhile is to fully zero-shot robotics: entire categories of tasks that a robot has never seen, executed at high success rates and high speeds, with no task-specific data at all. If the tasks are varied, complex, and valuable enough, this can be considered as requiring full physical AGI.

But there are also concrete milestones before that, which can build a progressive path: instead of fully zero-shotting, allow a small amount of robot data for a particular task — call it X — and execute that task at high levels of performance. Then the goal-driven roadmap becomes clear: keep decreasing X while pushing performance higher. For example, broadly achieving 99%+ success rates with roughly one hour of robot data, will have broad commercial viability. That is a concrete, measurable, goal-driven milestone that is independent of methods.

Also, as I’ve found before, choosing concrete, yet ambitious, goals in research is actually more productive as a springboard for branching out into a wider set of goals. Oddly, this can be even more productive than picking a method that feels like it could have a wide set of goals. Case in point: one of the first multimodal language models6 was created for a robotics-driven goal. It was, among other things, evaluated on medical benchmarks.7 This came out of a solve-whatever-is-needed mentality, not from hanging onto methods. Instead, being goal-driven affords you the agility to consider any method that gets you to your goal.

How far can we go?
Second, it is limiting to constrain machine learning via questions of “or” (e.g. choosing strictly between method A or method B). A deeper truth lies in asking “how far we can go?”, or even better, developing a deeper understanding of the objectives and constraints.

It is very natural to think that things must fit into categories, or that an approach or source must be “picked”. Every discipline can fall in this trap. To give some close-to-home examples, at previous points in robotics, the view has been that one must work on “perception or control”. Or another example is product managers at AI companies thinking in the early 2020s that every little application is destined to have their own specialized model, not realizing the benefits of vast cotraining.

But instead, the real question is, given what is achievable subject to the constraints, how far can we go? And which of the constraints can be removed? How far can we really go? To give one example, the Chinchilla8 paper was a truly lovely contribution that comes out of this type of thinking, one of those papers both celebrated at NeurIPS (Outstanding Paper) and with immediate massive impact in industry.

Most of the time, a question of “or” can be converted to a question of “and”, then to a question of “how much of each”, then eventually to a deeper question about the broader objectives and constraints.

Over the past two years, we have been revising our training methods with this philosophy in mind. For over a year, we have been experimenting with combining ideas from across what you might call VLAs, world models, and beyond. The more a model combines capabilities from different disciplines, the harder it is to categorize. And at the end of the day, what matters is: how far does it go?

Building for the world that’s coming
Third, the supply side will change. You have to think about not only the current constraints, but how those constraints will inevitably change. This is more important the faster the constraints are changing.

One current constraint, some say, is that there is not a lot of robotics data. This is not a long-term view. Now with over half a million hours of physical interaction data, we are able to ask questions without this constraint.

Similarly, a big part of the motivation for bringing vision-language training into robotics was that we didn’t have enough data inside robotics itself. So, in some sense, all of the vision-language training can be a helpful crutch while we don’t have enough robotics data. Sure there are more bytes of video that exist in the world than language, but still, it’s another crutch. What’s after the crutch? Will you still want the crutch?

Towards physical AGI
Goals are more powerful than methods, optimize given the constraints instead of picking lanes in categories, and those constraints themselves will inevitably change.

We’ve been committed to rethinking everything for physical AGI since day one of Generalist. This is what led to GEN-1, a model trained from scratch on our (world’s largest) dataset of physical interaction. Every aspect of the architecture, its training, and how inference is done was designed and iterated on without being constrained by decisions someone else made for a different purpose.

We’ve already shown glimpses of what it’s capable of — from scaling laws in robotics, generalizing to new environments and embodiments in hours, to improvisational intelligence emerging from large-scale pretraining. And this is just the beginning.

More soon.

1
GEN-1: Scaling Embodied Foundation Models to Mastery (Generalist, 2026)
2
Knowledge Distillation: A Good Teacher is Patient and Consistent (Beyer and Zhai et al., 2022)
3
RT-2: Vision-Language-Action Models (Brohan et al., 2023)
4
Video Language Planning (Du et al., 2023)
5
An Opinionated Guide to ML Research (Schulman)
6
PaLM-E: An Embodied Multimodal Language Model (Driess et al., 2023)
7
Med-PaLM M (Tu et al., 2023)
8
Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)

Instant Policy: In-Context Imitation Learning via Graph Diffusion (Vosylius & Johns, 2024)
10
Native Video-Action Pretraining for Generalizable Robot Control (Zhang et al., 2026)
11
Coarse-to-Fine Imitation Learning: Robot Manipulation from a Single Demonstration (Johns, 2021)
12
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., 2017)
13
Diversity is All You Need: Learning Skills without a Reward Function (Eysenbach et al., 2018)
14
GEN-0: Embodied Foundation Models That Scale with Physical Interaction (Generalist, 2025)
15
Scaling Laws for Neural Language Models (Kaplan et al., 2020)
16
GEN-1: Scaling Embodied Foundation Models to Mastery (Generalist, 2026)
17
Data Distributional Properties Drive Emergent In-Context Learning in Transformers (Chan et al., 2022)
18
Large Language Models as General Pattern Machines (Mirchandani et al., 2023)
19
One-Shot Imitation Learning (Duan et al., 2017)
20
The Dark Matter of Robotics: Physical Commonsense (Generalist, 2026)
21
On the Opportunities and Risks of Foundation Models (Bommasani et al., 2021)
22
The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (Akyürek et al., 2024)



Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Research
April 2, 2026
Generalist Team
14 min read
Table of Contents
Introduction
Scaling the Pretraining Era of Embodied Intelligence
Introducing GEN-1
Defining Mastery
Capabilities
Reliability
Speed
Improvisational Intelligence
Limitations
Rethinking Alignment for Embodied Intelligence
Looking Ahead
General Intelligence Born from the Physical World
Citation
More Videos
Back to Blog
GEN-1
Scaling Embodied Foundation Models to Mastery
We’ve created GEN-1, our latest milestone in scaling robot learning. We believe it to be the first general-purpose AI model that crosses a new performance threshold: mastery of simple physical tasks. It improves average success rates to 99% on tasks where previous models achieve 64%, completes tasks roughly 3x faster than state of the art, and requires only 1 hour of robot data for each of these results. GEN-1 unlocks commercial viability across a broad range of applications—and while it cannot solve all tasks today, it is a significant step towards our mission of creating generalist intelligence for the physical world.


At Generalist we are building towards physical AGI and making it useful to everyone. Today, we introduce our latest model, GEN-1. It is a large multimodal model that emits actions in real-time. It demonstrates several advanced capabilities compared to our previous models, and is a significant step towards our mission.

Five months ago, with GEN-0, we showed for the first time that scaling laws1 exist in robotics–bringing physical AI models into the pretraining era, which has analogously underpinned predictable progress in language models.2 GEN-0 was made possible by a new multimodal architecture trained on our own (world’s largest) robotics pretraining dataset, and it demonstrated the ability to quickly learn new tasks, adapt to new environments,3 and display moments of physical commonsense.4

Today, we announce GEN-1, which through further scaling of GEN-0’s foundation, and accelerated by algorithmic advances, is starting to show a significant shift in what these models can deliver. GEN-1 can begin to master simple tasks – on several tasks the model now exceeds 99% success rates (reliability), can complete tasks up to ~3x faster than the prior SOTA (speed), and exhibits a broad range of emergent behaviors to recover in unexpected scenarios (improvisation). In each case, these results require only approximately 1 hr of robot data.

We believe GEN-1 to be the first general physical AI model to cross a key threshold: unlocking commercial viability across a broad range of tasks—with a level of generality that is impossible to match with traditional automation, and at performance levels previously thought to be out of reach for robotics models. We previously created the first wave of embodied foundation models,5 including VLAs6 and world models,7 and we knew they were far from perfect. The progress of GEN-1 follows our own full redesign of embodied foundation models built for the real world, and is trained from scratch on our dataset of now half a million hours of real-world data.

GEN-1 represents a step change in capabilities, but it does not solve all tasks. It strengthens our view that continued scaling of our models with physical experience will continue to yield discoveries that unlock broader physical intelligence, expand the range of viable tasks, and open new application areas.

We are excited by these results, but we are still early in the journey. We believe the true nature of generalist intelligence involves the ability to achieve high levels of mastery across all physical work, and GEN-1 clarifies how we evaluate progress. GEN-1 shows early signs of new levels of mastery, which we define as the combination of reliability, speed and improvisation. Below, we detail these new capabilities from GEN-1, including videos of robots doing several different dexterous tasks hundreds of times in a row for hours.

Scaling the Pretraining Era of Embodied Intelligence
Previously, with GEN-0, we showed for the first time that scaling laws exist in robotics. Importantly, it demonstrated that it was possible to scale up robot learning in a generalized way – every zero-shot task we tracked would simultaneously improve. However, its performance was not sufficient to be used in commercial settings. Now with GEN-1, through further scaling of data and compute, and accelerated by algorithmic advances, we are starting to see some tasks cross the level of performance needed to be deployed in economically useful settings.

This parallels what has underpinned progress in large language models (LLMs) as they have been scaled over the past 8 years. GPT-28 showed a scalable path for multitask learning, but struggled to be deployed in economically valuable or useful software products. Scaling the model to GPT-39 showed the scaling laws held, new capabilities emerged, and the model became economically viable for certain tasks, such as copywriting for ads. As LLMs have scaled, each subsequent model generation has brought forth new capabilities that meet the performance requirements for a new set of tasks. Similarly, GEN-1 can begin to master simple tasks, but the more important concept supported by scaling is that we can expect each new generation of model to result in a new set of increasingly complex tasks that can be mastered.

Notably, this progression also validates the data engine behind these models. Previous general models in robotics that surpass 90% success have depended on enormous teleoperation datasets that are expensive and difficult to scale. Instead, for GEN-0 and GEN-1 the base foundation model is trained without any robot data—it instead uses data from low-cost wearable devices on humans doing millions of activities, and provides an existence proof that this pretraining can lead to high levels of mastery without requiring large teleoperation or simulation datasets.

Introducing GEN-1
GEN-1 comprises innovations across pre-training advances, post-training techniques, learning from experience (RL), multimodal human guidance, as well as new inference-time techniques. The pre-training advances have contributed to shifting the curve of compute efficiency of pretraining intelligence, and the others all contribute to unlocking higher performance for any given task. In addition to these advances, GEN-1 has also been scaled significantly since our previous model, GEN-0: this includes more compute and more data, trained on our dataset which now includes over half a million hours of high-fidelity physical interaction data.

While we may call GEN-1 a model, it is even more accurate to refer to GEN-1 as a system. Just as with frontier LLM chatbots and APIs, there are many system-level components across inference and model harnessing that critically advance its performance beyond being just a set of model weights.

GEN-1 is a data-efficient learner: in some tests, GEN-1 can achieve comparable performance to GEN-0 with 10x less task-specific data and fine tuning steps. Additionally, each of the results shown are built with only approximately one hour of robot data. The pretraining dataset contains no robot data, so when GEN-1 adapts to a new task, it is simultaneously adapting to that robot embodiment and to that task for the first time.

Defining Mastery
Embodied foundation models should be reliable, fast, and able to recover from unexpected scenarios. We use the term mastery to refer to the combination of all of: reliability, speed, and improvisational intelligence. While reliability and speed are more straightforward to measure, we believe it is improvisational intelligence that has most critically been missing from robotics before.

Reliability. The ability to reliably accomplish tasks is table stakes for real-world deployment. Traditional systems have performed repetitive motions reliably for decades, but this has evaded end-to-end robotics models. When high performance has been achieved, it is typically through resource-intensive teleoperation data on a specific system, limited to a narrow set of tasks, or achieved at the expense of complexity. The real challenge is not just achieving high performance once, but delivering robust, repeatable performance across tasks, systems, and environments.

Speed. Robotics has long suffered from a speed barrier: demo videos of dexterous general-purpose models are too slow. But breaking this speed barrier is not so simple. As speeds increase, the world becomes less quasi-static: velocity terms rise, friction dynamics change, motions blur, and there are increasing constraints on the precision, reactivity, and inference. What matters, too, is not how quickly the motors are moving, but how quickly the task is accomplished.

Improvisation. To thrive in unstructured environments, robots must have the ability to creatively improvise solutions in unexpected scenarios–to respond and adapt rather than rely on predefined behaviors. As we have previously discussed, we believe that physical commonsense is essential to achieving this type of freestyle problem solving. Without it, robots may execute routines well, but struggle when the world departs from the script.

Reliability and speed have been core to industrial robotics since the early 1960’s–but that history is built on precision and tightly controlling the robot’s environment, not intelligence. Instead, general physical AI models take a very different approach, via intelligence instead of restriction. As William James (late 19th century founding father of modern psychology) wrote, intelligence is the ability to reach the same goal by different means. Improvisational intelligence enables robots to thrive in unstructured environments, and also fuels better reliability and speed for generalist models.

When evaluating mastery, it is also essential to consider how much data is required to reach that performance for any given task.

Capabilities
Reliability
GEN-1 can perform several tasks at high levels of reliability over long durations without intervention. We show here 6 tasks: kitting auto parts for more than an hour, folding t-shirts 86 times in a row, servicing robot vacuums over 200 times in a row, packing blocks over 1,800 times in a row, folding boxes over 200 times in a row, and packing phones over 100 times in a row.


Kitting Auto Parts

Play Video

0:00

5:20

Kitting Auto Parts · 1x speed
1x speed, fully autonomous

Play Video

0:00

1:33

Kitting Auto Parts · 50x
One hour without intervention (50x)

T-Shirt Folding

Play Video

0:00

1:18

T-Shirt Folding · 1x speed
1x speed, fully autonomous

Play Video

0:00

3:14

T-Shirt Folding · 50x
86 times in a row without intervention (50x)

Servicing Robot Vacuum

Play Video

0:00

1:54

Servicing Robot Vacuum · 1x speed
1x speed, fully autonomous

Play Video

0:00

3:40

Servicing Robot Vacuum · 50x
200+ times in a row without intervention (50x)
Figure 1: Servicing Robot Vacuums. GEN-1 achieves 99% success rates, notably higher than GEN-0 (50%) or a from-scratch version of GEN-0 with no pretraining (2%).

Packing Blocks

Play Video

0:00

3:47

Packing Blocks · 50x
Packing blocks 1,800 times in a row without intervention (50x)

Folding Boxes

Play Video

0:00

5:29

Folding Boxes
200 times in a row without intervention
Figure 2: Folding Boxes. GEN-1 achieves 99% success rates, notably higher than either GEN-0 (81%) or a from-scratch version of GEN-0 with no pretraining (13%).

Packing Phones

Play Video

0:00

5:36

Packing Phones
100 times in a row without intervention
Figure 3: Packing Phones. GEN-1 achieves 99% success rates, notably higher than either GEN-0 (62%) or a from-scratch version of GEN-0 with no pretraining (42%). Note: these are apples-to-apples comparisons against our November 2025 version of GEN-0 models. In March 2025 at GTC, on a similar task, we showed a GEN-0 pretrained model with additional advances since November 2025.
Without pretraining, tasks trained from scratch exhibit very poor performance (average 19%). GEN-0 models finetuned on these tasks achieves better, but not production-ready success rates (average 64%), GEN-1 crosses into production-level success rates (average 99%).

Speed
These videos are at 1x speed and fully autonomous. They are not sped up:


Packing Phones
GEN-0
⏱️
0.0s
⏱️
0.0s
GEN-1

Folding Boxes
GEN-0
⏱️
0.0s
⏱️
0.0s
GEN-1
Figure 4: Speed comparisons for folding boxes, comparing GEN-1 to the previous SOTA. To simplify and enable a comparison, we only count the time spent folding the box, from the moment of touching it for the purpose of folding to the moment folding is complete. For the previous SOTA, both GEN-0 (source),10 and π0 (source)11 used identical boxes and took roughly 34 seconds, similarly to π*0.6 (source)12 on a comparable but distinct box. Instead, GEN-1 is 2.8x faster, able to achieve box folds in ~12 seconds.
On two challenging dexterous tasks, GEN-1 enables task completion speeds at roughly ~3x the state of the art. Importantly, GEN-1 can improve task completion speeds to be faster than demonstrations, and can react to new object physics at those speeds accordingly. GEN-1 can assemble a box in 12.1 seconds – this is 2.8x faster than prior SOTA (GEN-0 and π0 both took roughly ~34 seconds on identical boxes). GEN-1 can also pack a phone into a case in 15.5 seconds, at 2.8x the speed of GEN-0.

Several components enable these speed levels. For one, the models learn from experience to achieve these speeds. Additionally, GEN-1 introduces an evolution of the way we do inference with Harmonic Reasoning. Further, due to our data collection devices, the models have access to a wide array of pretraining data of completing various other tasks at high speeds (and thus transfer knowledge from general exposure to the dynamics involved), in contrast with traditional teleoperation systems that naturally produce slower, less fluid data due to the lack of force feedback, latency issues, and visibility challenges.

Improvisational Intelligence

We see a notable shift in how these models respond creatively to unexpected scenarios. In a long-horizon automotive kitting example, if a washer is bumped so far that it’s no longer held properly, the robot can either set it back down to regrasp it, or partially insert it into the slit to leverage extrinsic dexterity for regrasping, or even decide to use its other hand to enable bimanual in-hand regrasping. For the large deformable objects, if they end up in very unexpected configurations, the model figures out how to recover. These behaviors are well outside the training distribution, and directly contribute to recovering from unexpected long-tail events.

Limitations
GEN-1 is not without limitations. For instance, while we have shown several dexterous tasks at 99%+ success rates, not all tasks that we have attempted are able to hit these rates. Furthermore, some tasks would require even higher success rates or speeds to be useful in real settings. Nevertheless, we expect the next generation of models to unlock a broader range of more complex tasks that can be mastered, and we expect per-task data requirements to reduce over time as the base models improve.

Rethinking Alignment for Embodied Intelligence
One notable observation is that although pretraining on large-scale interaction data unlocks emergent improvisation (e.g. shaking a bag to seat an object, reorganizing misplaced items, or reaching for falling objects), these are physical actions with real consequences. The definition of success in robotics is not universal—it is task-specific, workflow-specific, and ultimately user-defined.13 It is not only about what the robot must do, but also (perhaps, more importantly) what it should not do. Hence, emergent behaviors can be a strength (e.g. recovery behaviors not explicitly trained for), but also at times a liability. As embodied foundation models grow to become more capable out of the box, we aim to improve our methods of alignment, and precisely steer them into delivering the behaviors that users actually want.

Looking Ahead
Building GEN-1 was not easy—we re-designed our distributed training infrastructure to support petabytes of physical interaction data as a first-class citizen. We spent months improving training stability, building custom kernels, inventing new forms of paged attention to enable real-time inference, honing post-training techniques (alongside foundations in theoretical RL and multimodal human guidance), and hardening controls to be even more smooth and precise. We designed new hardware and shipped thousands of robot hands across new geographies for exposure to unique physical activities. Nevertheless, we believe these advances will lay the groundwork for future research as we continue to scale our data engine into the next phase of capabilities.

General Intelligence Born from the Physical World
For us, GEN-1 is more than just a model. It captures an important part of artificial intelligence that we think is missing from the chatbots that we have today. It’s the intuition and open-ended problem-solving skills born from acting in the real world—combining knowledge that is grounded in real physics, with a deep understanding of how space and time matters, and that actions lead to consequences. It’s what affords the autonomy to recover from the unexpected (before it gets much worse), rather than having to be nudged along by a human every step of the way to avoid irreversible failures. For machines, we believe it is only through experiencing the physical world, that all the knowledge on Wikipedia can finally make sense.

We are still early in the journey, and we are excited about the next frontiers of embodied intelligence and beyond. Early access partners will have access to GEN-1 starting today. If you’d like to use our models, please email partnerships@generalistai.com. If you’re interested in joining us on our mission, please visit generalistai.com/careers.

Citation
Please cite this work as

Generalist Team, “GEN-1: Scaling Embodied Foundation Models to Mastery”, Generalist AI Blog, Apr 2026.
Or use the BibTeX citation:

@article{generalist2026gen1,
author = {Generalist Team},
title = {GEN-1: Scaling Embodied Foundation Models to Mastery},
journal = {Generalist AI Blog},
year = {2026},
note = {https://generalistai.com/blog/gen-1},
}
More Videos

















1
Scaling Laws in Robotics with GEN-0 (Generalist, 2025)
2
Scaling Laws for Neural Language Models (Kaplan and McCandlish et al., 2021)
3
The Real Breakthrough Behind Our GTC Demo (Generalist, 2026)
4
Physical Commonsense (Generalist, 2026)
5
PaLM-E: An Embodied Multimodal Language Model (Driess et al., 2023)
6
RT-2: Vision-Language-Action Models (Brohan et al., 2023)
7
Video Language Planning (Du et al., 2023)
8
Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
9
Language Models are Few-Shot Learners (Brown et al., 2020)
10
GEN-0 Box Folding Demo (Generalist, 2025)
11
π0: A Vision-Language-Action Flow Model for General Robot Control (Black et al., 2024)
12
π*0.6 Box Assembly (Hausman, 2025)
13
Inference-Time Policy Steering through Human Interactions (Wang et al., 2025)




Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Research
November 4, 2025
Generalist Team
12 min read
Table of Contents
Introduction
Surpassing the Intelligence Threshold
Scaling Laws for Robotics
Robotics is No Longer Limited By Data
Citation
Back to Blog
GEN-0 / Embodied Foundation Models That Scale with Physical Interaction
For years, foundation models in robotics have primarily used vision-language pretraining as the stepping stone towards scaling robotics, allowing us to transfer1 the benefits of semantic generalization from existing large multimodal models. But what’s been missing is how to effectively scale large multimodal model training in the domain of robotics itself—to establish scaling laws that corroborate the consistent (and predictable) improvement of robot intelligence with more compute & data, as has underpinned progress in other domains e.g. LLMs.2 This requires an architecture, training procedure, and data engine that pushes new sensorimotor capabilities, provides behavioral generalization, and grows with the vast and ever-expanding experience generated by interacting with the real physical world.

To this end, we’re introducing GEN-0, a new class of embodied foundation models built for multimodal training directly on high-fidelity raw physical interaction. Its architecture builds on the strengths of vision and language models while also going beyond them—natively designed to capture human-level reflexes and physical commonsense. One core feature is Harmonic Reasoning, in which the models are trained to simultaneously think and act seamlessly. We’ve shared a glimpse of the capabilities of early precursors in our prior videos,3 and today we are sharing that not only does GEN-0 have breakthrough fundamental capabilities, but these capabilities are scaling:

Surpassing the Intelligence Threshold – in an unprecedented high-data regime for robotics, we observe a phase transition at 7B where smaller models exhibit ossification,4 while larger ones continue to improve. We’ve since scaled GEN-0 to 10B+ model sizes, and observe fast adaptation to new tasks with increasingly less post-training.
Scaling Laws – GEN-0 models exhibit strong scaling laws, in which more pretraining data and compute consistently (and predictably) improve downstream post-training performance of the model across many tasks.
Harmonic Reasoning – Although for language chatbots it is straightforward to spend more time thinking before responding,5 the same is not as simple for physical systems acting in the real world – physics doesn’t stop. To address this problem, Harmonic Reasoning involves a fundamentally new approach to training models, and creates a “harmonic” interplay between asynchronous, continuous-time streams of sensing and acting tokens. This allows us to scale to very large model sizes without depending on System1-System2 architectures6 or inference-time guidance.7
Cross-Embodiment – GEN-0 architecture works on different robots by design. We have tested our models on 6DoF, 7DoF, and 16+DoF semi-humanoid robots.
No Longer Limited By Data – GEN-0 is pretrained on our in-house robotics dataset, which includes over 270,000 hours of real-world diverse manipulation data, growing at a rate of 10,000 hours a week and accelerating.
The Science of Pretraining – different mixtures of pretraining data (from various sources e.g. data foundries) yield GEN-0 models with different characteristics. We share some early notes from our empirical observations in this high-data regime, and how that traces back to specific data collection operations.
We believe that GEN-0 marks the beginning of a new era: embodied foundation models whose capabilities predictably scale with physical interaction data – not just from text, images, or simulation – but the real world. Here are videos of GEN-0 in action on new tasks:

Build a camera kit (top view). This is a long horizon dexterous task that involves placing a cleaning cloth into a box, folding in a cardboard tray, picking up a camera and unsheathing it from a plastic bag, placing it into the box, closing the box (and inserting the tiny flap), then discarding the plastic bag. The model does not maintain any explicit notion of a subtask, and performs this all within a single stream of harmonic reasoning.
Surpassing the Intelligence Threshold
Our scaling experiments show that GEN-0 models must be large enough to absorb vast amounts of physical interaction data. We observe that smaller models exhibit a phenomenon similar to ossification4 under data overload, while larger ones continue to improve—demonstrating a surprising “phase transition” in the intelligence capacity of our models:

1B models struggle to absorb complex and diverse sensorimotor data during pretraining – model weights become unable to absorb new information over time.
6B models begin to benefit from pretraining and show strong multi-task capabilities.
7B+ models are able to internalize large-scale robotic pretraining data that transfers to downstream tasks with only a few thousand steps of post-training.
Figure 1: scaling GEN-0 model size improves performance
Figure 1. Scaling GEN-0 model size (different colors) improves performance in terms of next-action validation prediction error (y-axis, lower is better) on a completely-withheld (i.e. zero-shot) long-horizon downstream task. 1B parameter models exhibit clear and early ossification, while 6B and 7B models perform better at absorbing pretraining respectively. The x-axis is pretraining compute normalized so that GEN-0 7B is 1.0.
To our knowledge, this is the first time that model ossification8 has been observed in robotics. This might have eluded past research due to (a) the lack of a high data regime in robotics until now, and (b) large enough model sizes in this regime. Ossification has previously been observed in LLM literature49 in the high data regime but with much smaller models, on the order of O(10M) parameters rather than O(1B). The observation that this phase transition occurs in robotics but with much larger model sizes echoes Moravec’s Paradox:10 what humans find effortless—perception and dexterity—demands far more computational complexity than abstract reasoning. Our experiments suggest that intelligence in the physical world (i.e. physical commonsense) may have a higher activation threshold in terms of compute, and we’re only beginning to explore what lies beyond.

Scaling Laws for Robotics
Scaling laws are commonly measured during pretraining, as shown in Figure 1, which shows the relationship of model size and compute on a downstream zero-shot task during pretraining. Another type of scaling law relates to the benefits of pretraining that persist into finetuning.4 At sufficient model scale, we also observe a strong power-law relationship (Figure 4) between pretraining data scale and downstream post-training performance. This applies to all of our tasks we’ve measured, including partner and customer-inspired applications and their workflows across a wide range of industrial sectors – including apparel, manufacturing, logistics, automotive, and electronics.

More specifically, we take a variety of model checkpoints (Figure 2) that have been pretrained using our training procedure on different subsets of our pretraining dataset, and then post-train these checkpoints on multi-task language-conditioned data i.e. supervised fine-tuning simultaneously on 16 different task sets. We find that more pretraining improves downstream model performance across all tasks (Figure 2).

Validation loss scaling across pretraining data sizes
Next-action prediction error across 16 task sets
Figure 2. With increasingly more pretraining data (different colors), multi-task model performance during post-training improves in terms of validation loss (top) as well as next action prediction error (bottom 4x4 grid) across all 16 task sets. These tasks include ones that evaluate dexterity (e.g. build Lego), industry-specific workflows (e.g. fast food packing), and generalization (e.g. “_ anything” tasks).
We also observe that these trends transfer to real-robot performance, measured with blind A/B evaluations. Increasing the amount of pretraining data improves downstream task success rates (Figure 3) using closed-loop policies with models post-trained on only 5.6 hours of task-specific data. While the gains are significant in this low-data regime, the highest task success rates (up to 99% peak performance in certain cases) emerge when large-scale pretraining is paired with ample task-specific post-training data. For valid comparisons, we ensured no overlap between the pretraining and post-training datasets, which were collected by different people in entirely different environments.

Real-robot task success rate vs. pretraining data
Figure 3. For real-robot evaluations, more pretraining data (colors denote different pretrained model bases) yields higher downstream average task success rates via closed-loop policy rollouts (performance is plotted with standard error). Models on the left are post-trained on 5.6 hours (1%) of task-specific data, and the best models (right) use both the full pretraining dataset and all (550+ hours) of available task-specific post-training data.
Model performance is predictable with a power-law relationship (Figure 4), with which we can answer questions like “how much pretraining data do we need to reach a specific next-action prediction error?” or “how much post-training data (for a specific task) can we buy with more pretraining data?” Given a fixed data and finetuning budget on a downstream task, and given a pretraining dataset of varying size 
, the validation error 
 on the downstream task can be predicted via a power-law of the form:

For example, in the case of Clothes Handling (which involves sorting, unscrambling, buttoning, and hanging clothes in a real workplace), we can predict model performance given 1 billion action trajectories. These estimates guide conversations on partner-related tasks and can provide estimates on how much more data is needed to reach specific levels of performance.

Scaling law for Clothes Handling task
Figure 4. Our scaling laws provide a good description for asymptotic next action prediction error on a post-trained model for a given task set as a function of pretraining dataset size (in terms of number of action trajectories). Together with model size scaling laws, we can use these results to predict optimal allocation of pretraining compute and data for any downstream post-training task.
Robotics is No Longer Limited By Data
Our Foundation models are trained on an unprecedented corpus of 270,000 hours of real-world manipulation trajectories collected across diverse activities in 1,000s of homes, warehouses, and workplaces worldwide. Today, our robot data operations provide over 10,000 new hours per week and are accelerating. This is all powered by a global network of hardware and 1,000s of data collection devices and robots.

GEN-0 dataset size compared to other large robotics datasets
Figure 5. GEN-0 is trained on orders of magnitude more real-world manipulation data than some of the largest robotics datasets that exist to date (as of Nov 2025).
Mapping the Universe of Manipulation
To scale GEN-0 capabilities, we are constructing the largest and most diverse real-world manipulation dataset ever built, including every manipulation task humans can think of – from peeling potatoes, to threading bolts – spanning homes, bakeries, laundromats, warehouses, factories, and more. Here is an example internal search tool we have built to explore this universe:

Figure 6. This is an example of searching through <1% of our pretraining dataset, which includes manipulation data from millions of diverse activities across different environments. The visualization navigates the user through a t-SNE map of corresponding language label embeddings in the dataset. Given a text description, the visualizer locates the nearest neighbor region, and randomly samples in the area a collection of related videos and displays them.
Infrastructure for Internet-Scale Robot Data
Building the operations and ML infrastructure to support this is no easy feat. For robot models and data at this scale, we built custom hardware, dataloaders, and network infrastructure (including laying new dedicated Internet lines) to support the uplink bandwidth from a diverse set of data collection sites all around the world. We’ve negotiated multi-cloud contracts, built custom upload machines, scaled to O(10K) cores for continual multimodal data processing, compressed dozens of Petabytes of data, using dataloading techniques behind frontier video foundation models, capable of absorbing 6.85 years of real-world manipulation experience per day of training.

Science of Pretraining
From large-scale ablations, we find that data quality and diversity matters more than sheer volume, and that carefully constructed data mixtures can lead to different pretrained model characteristics. For example, Table 1 shows the performance metrics of different models trained on 8 different pretraining datasets, and their downstream impact when finetuned on 10 long-horizon task sets, organized into 3 groups that evaluate different dimensions: dexterity, real-world applications, and generalization.

Performance is measured in terms of validation prediction M.S.E. 
 and reverse Kullback–Leibler divergence11 (reverse KL), which better measures mode-seeking behavior.1213 To estimate reverse KL, we use a Monte-Carlo estimator where the policy induces an empirical density 
 via a unit-variance mixture of Gaussians centered at 
 policy samples 
, and the data/ground-truth induces a unit-variance Gaussian 
 centered at 
. We approximate the expectation with policy samples:

Experiments show that models with both low prediction errors and low reverse KL tend to perform better with supervised finetuning (SFT) for postraining, while models with high prediction errors and low reverse KL tend to be more distributionally multimodal, which can help post-training reinforcement learning. Having multiple data collection strategies at scale allows us to continually A/B test which data improves pretraining the most.

Partner & Class (Pred Err)	Dexterity	Applications	Generali­zation
Partner A Class 1	0.00307682	0.00334155	0.00308992
Partner A Class 2	0.00306196	0.00333253	0.00306503
Partner A Class 3	0.00305728	0.00331309	0.00305888
Partner A Class 2 + 3	0.00315980	0.00341899	0.00315661
Partner B Class 1	0.00302728	0.00330365	0.00304627
Partner B Class 2 Objs	0.00314415	0.00341147	0.00315975
Partner B Class 2 Skills	0.00301995	0.00329235	0.00305292
Partner C Class 3	0.00306247	0.00332128	0.00307944
Partner & Class (Rev KL)	Dexterity	Applications	Generali­zation
Partner A Class 1	0.00200585	0.00258898	0.00198088
Partner A Class 2	0.00188744	0.00244642	0.00193866
Partner A Class 3	0.00198332	0.00246089	0.00190205
Partner A Class 2 + 3	0.00184110	0.00228588	0.00185473
Partner B Class 1	0.00189286	0.00246051	0.00192307
Partner B Class 2 Objs	0.00184719	0.00233209	0.00186721
Partner B Class 2 Skills	0.00182561	0.00242293	0.00190308
Partner C Class 3	0.00192134	0.00236901	0.00190956
Table 1. These experiments compare different pretraining datasets, collected together with multiple data foundry partners, split across different classifications (i.e. modes) of data collection. Class 1 involves data on specific tasks, Class 3 involves do-anything type data, and Class 2 is everything in between. Different partners also have different operations, and we can use these experiments to evaluate between partners to iterate and provide feedback on what data to collect, how to do it, and which methods improve models the most.

More on these learnings in future posts.

Citation
Please cite this work as:

Generalist Team, “GEN-0: Embodied Foundation Models That Scale with Physical Interaction”, Generalist AI Blog, Nov 2025.
Or use the BibTeX citation:

@article{generalist2025gen0,
author = {Generalist Team},
title = {GEN-0: Embodied Foundation Models That Scale with Physical Interaction},
journal = {Generalist AI Blog},
year = {2025},
note = {https://generalistai.com/blog/gen-0},
}
1
PaLM-E: An Embodied Multimodal Language Model (Driess et al., 2023)
2
Scaling Laws for Neural Language Models (Kaplan and McCandlish et al., 2021)
3
Research Preview (Generalist, 2025)
4
Scaling Laws for Transfer (Hernandez et al., 2021)
5
Learning to reason with LLMs (OpenAI, 2024)
6
Helix: A Vision-Language-Action Model for Generalist Humanoid Control (Figure, 2025)
7
Real-Time Execution of Action Chunking Flow Policies (Black et al., 2025)
8
Note that in the LLM literature this phenomenon has been used to refer to the pretrain-to-finetune setting, whereas in our experiments (Figure 1) we observe ossification-type behavior of zero-shot generalization during the pure pretraining phase.
9
Overtrained Language Models Are Harder to Fine-Tune (Springer et al., 2025)
10
Mind Children (Moravec, 1988)
11
Divergence measures and message passing (Minka, 1988)
12
A Divergence Minimization Perspective on Imitation Learning Methods (Seyed Ghasemipour et al., 2019)
13
Imitation Learning as f-Divergence Minimization (Ke et al., 2020)


Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Research
July 23, 2026
Generalist Team
5 min read
Table of Contents
Introduction
Generalizing GEN-1 to Many Hands
Every Hand Is a Different Language
Studying How GEN-1 Adapts to New Hands
Adapting to Different Hands On-the-Fly
Towards a Cambrian Explosion of Robot Form Factors
Citation
Back to Blog
GEN-1,1 our latest embodied foundation model, now supports a broad range of robot end effectors — from five-finger anthropomorphic hands, to specialized tools with new modes of actuation, and everything in between. By training GEN-1 to work with these new hands, we demonstrate that a single base model can learn sensorimotor policies on robots that transfer across radically different ways of interacting with the physical world. Below, we walk through examples of these hands in action and where we think this is headed.


Play Video

0:00

2:09


Towards Machines with a Thousand Hands
Watch on Youtube
Generalizing GEN-1 to Many Hands
GEN-1 is pretrained on our in-house robotics dataset, which now spans a wide variety of different end effectors across more than half a million hours of real interaction data. Some end effectors involve new form factors with their own actuation schemes and camera positions, a few inspired by real commercial use cases. Others are off-the-shelf tools, printed parts, or custom modifications to our standard two-finger grippers — approximately 9,000 variations so far — all chosen to expose the model to a broad range of contact physics.

Hands can change, physics does not.
Each end effector is a different sensorimotor interface through which GEN-1 experiences the physical world — a way to learn about geometry, contact, friction, forces, and dynamics. Scaling pretraining across thousands of these interfaces teaches GEN-1 universal sensorimotor representations: a general physical commonsense2 that transfers to new hands and new ways to grasp, push, pull, twist, and more.


0:00

2:19


Download Video
Robot uses extra wide hands with thin flat fingers to move boxes.

0:00

1:30


Download Video
Robot uses a power screwdriver to tighten four bolts.

0:00

1:04


Download Video
Robot uses tongs to place eggs into a pot of water.

0:00

1:17


Download Video
Robot uses a whisk to mix baking ingredients.
Every Hand Is a Different Language for Physical Interaction
Each hand is its own vocabulary for acting in the world. Power screwdrivers rotate faster than fingers can match. Controlling a tape dispenser requires managing tension and placement at once. Tongs introduce compliance and spring-force dynamics that change how an object should be approached. Metal spatulas and scrapers work against a surface rather than around an object, which means reasoning about distributed contact rather than single points of contact. A box cutter or a vegetable peeler demands controlled force along a constrained path.

Just as training on multiple languages produces more capable language models3 (concepts learned in one language improve understanding in another), learning across many hands stands to benefit physical intelligence. A model trained across embodiments can gather shared knowledge across instances and more readily separate what is specific to a tool from what is universal about the world. Switching between end effectors to reach a goal then becomes a form of physical reasoning: using the right tool for the right job, much as multilingual chain-of-thought4 can improve language model reasoning for downstream reinforcement learning.


0:00

2:29


Download Video
Robot uses various tools to sweep and scrape debris off the table.
Not every hand will help, though. Some end effectors may provide little learning signal — and some, like our two-finger grippers, will carry more real-world weight than others, much as English dominates today’s language models. We're studying how each new end effector shifts the pretrained model, expanding the dataset deliberately, and evaluating on real benchmarks.

Studying How GEN-1 Adapts to New Hands
Hands and tools are not just different shapes; each one is a different form of interaction the model needs to learn. We can measure this by analyzing how much model weights shift during fine-tuning as a quantitative signal of how novel an end effector really is. Comparing pretrained GEN-1 to the same model after fine-tuning on a new tool, we can treat the difference as a “task update” 
, where 
 is the pretrained model and 
 is the fine-tuned model. Similar task vector5 analyses have studied how fine-tuning moves foundation models through weight space; here we apply the lens to embodiments, decomposing updates across sensor processing, harmonic reasoning6, and actuation in the architecture and visualizing them below.

Screwdriver
Tape Dispenser
Bottle Opener
Extra Wide Hands
Scrape + Sweep
Box Cutter
Whisk
Peeler
Tongs

Pretrained Base

Overlay All
GEN-1
GEN-1 Pretrained Base
0% change in model weights
GEN-1 model weight updates with different end effectors range from 2.5% to 11.4% in relative parameter norm. Controlling a power screwdriver, it turns out, takes more re-education than controlling a pair of tongs.
This tells us both which end effectors carry the most new information and where that novelty lands in the architecture. Whisks, for instance, shift the sensor-processing weights far more than peelers do, likely because the model has to perceive the whisk’s thin wire geometry. That points straight to a data intervention: collect more thin, visually sparse tools to precisely strengthen that subsystem.

Adapting to Different Hands On-the-Fly
What happens when you change the hand mid-task? We can test this directly: with the model mid-rollout, we physically swapped its end effector and let the same model keep running. It perceives the new tool, conditions on what it sees, and finds a new trajectory and contact strategy to reach the same goal.


Play Video

0:00

1:18


Mid-task End Effector Swap
Download Video
GEN-1 adapts to new manipulation strategies when its end effector gets changed mid task.
This works because training on mixed data forces the model to condition its behavior on the hand in front of it. Rather than memorizing a fixed set of manipulation strategies, it learns something closer to a prior over how shapes and contact surfaces interact with the world — and which actuation strategy should follow. The result is a single model that recognizes its own tooling and adapts: towards one intelligence for many hands.

Towards a Cambrian Explosion of Robot Form Factors

Nature didn’t converge on just a single solution for manipulating the physical world.
It exploded into millions. You can see this everywhere in the diversity of life around us: from the beak of a bird, to the trunk of an elephant, from the suction pads of an octopus, to the pollen baskets of a honeybee. Each gives its owner a different way to grasp, push, and pull, and these are only a few points in a design space so large we’ve barely begun to map.

Robots aren’t locked into the hands they’re born with.
Unlike humans, robots can easily swap mechanical hands, and tool changers7 are already common in automation. Powered by the right intelligence, robots can switch between end effectors and compose actuation modes to perform tasks in ways that go beyond what humans can do. Sometimes a highly specialized tool is exactly right, and Generalist systems should be capable of knowing when and how to use the best tool for the job.


Building General Physical Intelligence
If we can build general intelligence that understands the underlying physics of interaction, then the shape of the hand becomes secondary to the intelligence that drives it. A suction pad, a gripper, a brush, a plasma welding nozzle — are all just different interfaces through which the same intelligence can reshape the physical world.

The future of robot hands won’t look like ours. It will look more like a toolbox with a thousand hands: augmented, recombined, and scaled. Five-fingered hands will be one tool among many; limiting robots to only that would be a failure of imagination. Robots were always meant to extend what humans can do — to empower people to shape the physical world in places and at scales we never could before.

Citation
Please cite this work as

Generalist Team, “Towards Machines with a Thousand Hands”, Generalist AI Blog, Jul 2026.
Or use the BibTeX citation:

@article{generalist2026thousandhands,
author = {Generalist Team},
title = {Towards Machines with a Thousand Hands},
journal = {Generalist AI Blog},
year = {2026},
note = {https://generalistai.com/blog/towards-machines-with-a-thousand-hands},
}
1
GEN-1: Scaling Embodied Foundation Models to Mastery (Generalist, 2026)
2
The Dark Matter of Robotics: Physical Commonsense (Generalist, 2026)
3
Unsupervised Cross-lingual Representation Learning at Scale (Conneau et al., 2020)
4
Language Models are Multilingual Chain-of-Thought Reasoners (Shi et al., 2022)
5
Editing Models with Task Arithmetic (Ilharco et al., 2023)
6
GEN-0: Embodied Foundation Models That Scale with Physical Interaction (Generalist, 2025)
7
Automatic Tool Changer (Wikipedia)
Postscript
Recommended reading: The Beetle Roboticists: A Parable (Mason, 2024)


Index
Blog
About
Careers
Contact
YouTube
LinkedIn
x.com
© 2026
Generalist AI, Inc.
All rights reserved.
Story
March 24, 2026
Generalist Team
4 min read
Table of Contents
Introduction
A step-change in the speed of generalization
The timeline leading up to GTC
Generalization for free
A future where robots show up and just work
Back to Blog
The Real Breakthrough Behind Our GTC Demo
We ran a live demo at GTC last week, but the real story is how quickly we got it running.

GEN-0 has given us a step-change in the speed of generalization
Last week at NVIDIA GTC, we did a live demo of our GEN-0 foundational model. This was our first public live demo! We ran it nonstop during all open hours of the conference. The energy at the booth was awesome. We were amazed at how actively people sought us out and how quickly word spread. Thank you to all who stopped by!

But for us, what’s most notable is not the demo itself but how quickly we were able to put it together. GEN-0 models have proven to be so capable at generalizing to new robots and new environments, that we said yes to doing a demo knowing that we’d only have a few days to prepare with an entirely new type of robot. This would have been impossible even just a few months ago.

The timeline leading up to GTC: brand new robot, only in our office for a handful of days
We were fortunate to be asked by Universal Robots, the world’s #1 cobot manufacturer by volume, to join them in their booth to live demo our GEN-0 model on their new mobile manipulation platform.

This new mobile manipulation platform—consisting of UR7e arms on a MiR base and mounted together by a Vention frame—did not previously exist!

Despite not having ever seen or touched this robot in person, only about a month before the conference, we said yes to doing the demo.

Due to shipping time, robot setup, and other delays, in truth we only ended up having a small number of days to set up and prepare the demo.

Here’s the robot the day it showed up in our Boston office.
The robot the day it showed up in our Boston office
Two days later, running the demo task in our office.
This is a challenging multi-step task that emphasizes motion precision (tight tolerances of the components of the box) as well as force precision (it is especially challenging to not crumple the different paper and cardboard components).

We then shipped the robot to our SF office, where we had everything running within a day of arrival and just three full work days to do final prep of the demo before sending it to GTC.

First box packed in SF, within a day of the robot’s arrival.
First box packed in SF, within a day of the robot’s arrival
Generalization for free
At GTC, we unpacked the robot, booted it up, and its performance was identical to operating in our offices. Of course, since there wasn’t time, we used no data from inside the GTC exhibition hall.

First-run of the system in the UR booth at GTC.
The demo ran for every hour of the exhibit (no limited or scheduled times) and we brought out the hockey stick to demonstrate the model’s resilience.

A future where robots show up and just work
We took a robot platform that didn’t exist and had a live, public demo of GEN-0 running on the system within a handful of days.

This is a preview of a world where robots can just show up and work. It validates the power of GEN-0 as a true foundational model enabling things that were simply not possible before.

Reach out to partnerships@generalistai.com if you are interested in working together!

