### 引言

1997 年 5 月 11 日，国际象棋世界冠军加里·卡斯帕罗夫（Garry Kasparov）放弃了与 IBM 深蓝超级计算机的六场比赛中的最后一场比赛，深蓝以3.5–2.5胜出，标志人类历史上第一个世界冠军级别的棋类计算机的诞生。它是一个庞大的并行系统，通过树搜索进行国际象棋游戏的博弈，平均每秒钟可以搜索1亿次棋局。

- 搜索容量大。 以前对博弈树搜索的研究通常涉及搜索位置比深蓝少几个数量级的系统
- 硬件评估。 Deep Blue 评估功能在硬件中实现
- 混合软件/硬件搜索。 深蓝搜索结合了软件搜索，在通用 CPU 上编译的 C 代码实现，硬件搜索，在国际象棋芯片上的硅中编码
- 大规模并行搜索。 Deep Blue 是一个大规模并行系统，有超过 500 个处理器可用于参与博弈树搜索

之后是2016年，Stockfish 荣获顶级国际象棋引擎锦标赛（TCEC）世界冠军 。

2017 年 5 月 27 日，围棋人工智能AlphaGo参加了最后一场人机比赛，以3：0战胜当时排名世界第一的围棋世界冠军柯洁。AlphaGo的多次战绩足以证明其水准已经超过人类职业围棋的顶尖水平。AlphaGo训练与运行策略主要分以下几步：

- 行为克隆（behavior cloning），学习人类高水平棋手的上万局对弈来初始化策略网络（policy network）
- 策略网络进一步训练：通过策略网络的相互对弈，利用策略梯度训练
- 价值网络（value network）：在训练好的策略网络基础上训练
- 执行：利用训练好的策略网络与价值网络，执行蒙特卡洛树搜索（MTCS）

之后在AlphaGo的基础上诞生的AlphaGo Zero，以100:0的战绩击败了AlphaGo。AlphaGo Zero是一种白板式的新型AI，不依靠学习人类的知识，仅利用基础规则，通过自我对弈便能达到很高的水准。AlphaGo Zero有很多优点：

- 结构简单：不同于AlphaGo，AlphaGo Zero只由神经网络和蒙特卡洛树搜索算法组成
- 泛用性广：AlphaGo Zero的原理与算法可以应用到任何一种双人零和完美游戏当中
- 棋力更强：AlphaGo Zero在与别的棋类引擎对弈时有压倒性优势
- 上限更高：AlphaGo Zero在训练30天后超越 AlphaGo Master, 在训练40天后超越 AlphaGo Master 300 分。

强化学习是一种机器学习方法，通过与环境（Environment）交互获得的奖励（Reward）来指导智能体（Agent）在特定状态（State）下的行为（Action），使智能体获得最大的奖励或达成特定的目标。强化学习中的策略函数与价值函数都是高度复杂的非线性函数，因此可以结合深度学习模型，用神经网络去逼近上述函数，从而达到更好的效果，这就是深度强化学习方法。根据确定动作的具体方法，可以将深度强化学习分成价值学习与策略学习，它们分别用神经网络模拟价值函数与策略函数。在策略学习中同时用神经网络模拟价值函数，就形成了Actor-Critic方法。在AlphaGo与AlphaGo Zero中，确定动作时没有直接使用策略函数，而是采用了蒙特卡洛树搜索算法，进一步提升了模型性能。

### 中国象棋与围棋的不同

从两种棋类游戏本身的特点来看，两者有比较大的差异。首先是棋子的区别，围棋的棋子没有兵种之分，每个子的价值都相同，只用于区分黑白双方；而象棋的棋子分为7种兵种，不同种类的棋子功能、数量与重要性也不同。第二个区别是两者棋面大小区别，围棋的棋面为 $19\times19$ ，象棋的为 $9\times10$ 。第三个区别是行子的方式不同，围棋的棋子会越下越多，且棋子不能移动；而象棋的棋子有吃子的形式，会越下越少，棋子也可以移动且不同种类的棋子移动方式不同。第四个区别是规则的区别，围棋与中国象棋有各自不同的规则。除了上述区别之外还有一些别的区别，此处就不一一赘述了。

在算法层面上，首先两者的特征平面不同，除了棋盘大小的区别外，围棋的棋局只需要黑棋与白棋两个特征平面即可确定，而象棋有7种棋子，如果采用one-hot编码，每个棋局就需要14个特征平面。两者适应的网络结构也有很大差异，围棋的游戏规则是平移不变的，还具有旋转与反射对称性，与卷积神经网络的权值共享等结构相匹配，并且支持数据增强；而象棋的规则与位置有关，是非对称的，比围棋更加复杂，需要进一步改进网络结构。

### 我们改善了哪里

首先，是关于数据结构的更新，象棋与围棋相比具有更加多样的特征，我们可以开发出基于象棋棋盘意外的更丰富的数据结构以更加全面的捕捉与表示棋盘信息，比如：

- 当前棋盘矩阵与之前的 $N$ 个棋盘矩阵
- 目前双方的吃棋状态
- 最近几步的走棋信息
- ”将军“的次数等

但是不可否认的是，数据结构的丰富与我们的初衷有所背离，因为这意味着我们引入了人类的知识，而我们预想的是在训练期间不使用除了规则意外的任何人类知识，但是在文章《Mastering the Game of Go without Human Knowledge》中，作者提到了并不是所有的人类知识是有害的，如果我们控制好人类知识在模型训练过程中的“参与度”，是很有可能在提高训练效率，降低硬件配置的同时，提高其智能度的。

其次，模型训练中self-play的环节非常重要，那么有没有可能在这个环节中引入可能的创新呢？我们认为可以从self-play的引导机制入手。==在alpha go与alpha zero中，引导机制均使用了蒙特卡洛树搜索（只是alpha zero在真正下棋的过程中没有使用MCTS rollouts），考虑到象棋的搜索空间明显小于围棋，我们可以尝试通过自定义给予蒙特卡洛树的剪枝搜索达到较深的深度，以取得更好的效果。==

### Introduction

On May 11, 1997, world chess champion Garry Kasparov (Garry Kasparov) gave up the last of six matches against IBM's Deep Blue supercomputer. Deep Blue won 3.5–2.5, marking human history. The birth of the world's first chess computer of world champion level. It is a huge parallel system that performs chess games through tree search, which can search 100 million chess games per second on average.

- Large search capacity. Previous work on game tree search typically involved systems where the search positions were orders of magnitude fewer than Deep Blue
- Hardware evaluation. Deep Blue evaluation functionality implemented in hardware
- Hybrid software/hardware search. Deep Blue Search combines software search, implemented in C code compiled on a general-purpose CPU, with hardware search, coded in silicon on a chess chip
- Massively parallel search. Deep Blue is a massively parallel system with more than 500 processors available to participate in game tree search

Then in 2016, Stockfish won the World Championship of the Top Chess Engine Championship (TCEC).

On May 27, 2017, Go artificial intelligence AlphaGo participated in the last human-machine match and defeated Ke Jie, the then world No. 1 Go world champion, 3:0. AlphaGo's multiple records are enough to prove that its level has surpassed the top level of human professional Go. The AlphaGo training and operation strategy is mainly divided into the following steps:

- Behavior cloning, learning tens of thousands of games played by human high-level chess players to initialize the policy network
- Further training of the strategy network: through the mutual game of the strategy network, use the strategy gradient training
- Value network: trained on the basis of the trained policy network
- Execution: use the trained policy network and value network to perform Monte Carlo tree search (MTCS)

AlphaGo Zero, which was born on the basis of AlphaGo, defeated AlphaGo with a record of 100:0. AlphaGo Zero is a new type of whiteboard AI. It does not rely on learning human knowledge, but only uses basic rules to achieve a very high level through self-play. AlphaGo Zero has many advantages:

- Simple structure: Unlike AlphaGo, AlphaGo Zero is only composed of neural network and Monte Carlo tree search algorithm
- Wide versatility: the principles and algorithms of AlphaGo Zero can be applied to any two-player zero-sum perfect game
- Stronger Chess: AlphaGo Zero has an overwhelming advantage when playing against other chess engines
- Higher upper limit: AlphaGo Zero surpassed AlphaGo Master after 30 days of training, and surpassed AlphaGo Master by 300 points after 40 days of training.

Reinforcement learning is a machine learning method that guides the behavior (Action) of the agent (Agent) in a specific state (State) through the reward (Reward) obtained by interacting with the environment (Environment), so that the agent can obtain the maximum reward or achieve a specific goal. Both the policy function and the value function in reinforcement learning are highly complex nonlinear functions. Therefore, the deep learning model can be combined with the neural network to approximate the above functions, so as to achieve better results. This is the deep reinforcement learning method. According to the specific method of determining the action, deep reinforcement learning can be divided into value learning and policy learning, which use neural networks to simulate the value function and policy function respectively. Simultaneously using the neural network to simulate the value function in policy learning forms the Actor-Critic method. In AlphaGo and AlphaGo Zero, the strategy function is not directly used when determining the action, but the Monte Carlo tree search algorithm is used to further improve the performance of the model.

### The difference between Chinese chess and Go

Judging from the characteristics of the two chess games themselves, the two are quite different. The first is the difference between the chess pieces. The pieces of Go have no division of arms. Each piece has the same value and is only used to distinguish black and white. The pieces of chess are divided into 7 types of arms, and the functions, numbers and importance of different types of pieces are also different. . The second difference is the size of the two chess faces. The chess face of Go is $19\times19$ , and the chess face is $9\times10$ . The third difference is that the way of moving the pieces is different. The pieces in Go will move more and more, and the pieces cannot be moved; while the pieces in chess have the form of capturing pieces, and they will move less and less. The pieces can also move and different types of pieces move. different ways. The fourth difference is the difference in rules. Go and Chinese chess have different rules. In addition to the above differences, there are some other differences, which will not be repeated here.

At the algorithm level, first of all, the feature planes of the two are different. In addition to the difference in the size of the chessboard, the chess game of Go only needs two feature planes of black and white to determine, while chess has 7 kinds of pieces. For coding, 14 feature planes are required for each chess game. The network structures adapted by the two are also very different. The game rules of Go are translation invariant, and it also has rotational and reflection symmetry, which matches the structure of weight sharing of convolutional neural networks, and supports data augmentation; while chess The rules are related to position, are asymmetric, and are more complex than Go, and the network structure needs to be further improved.

### Where we improved

First of all, it is about the update of the data structure. Compared with Go, chess has more diverse features. We can develop a richer data structure based on the chess board to capture and represent the board information more comprehensively, such as:

- the current checkerboard matrix and the previous $N$ checkerboard matrices
- The current state of the two players
- Move information of the last few steps
- The number of "generals" etc.

But it is undeniable that the enrichment of data structures is a departure from our original intention, because it means that we introduce human knowledge, and we envision not using any human knowledge other than rules during training, but in the article In "Mastering the Game of Go without Human Knowledge", the author mentioned that not all human knowledge is harmful. If we control the "participation" of human knowledge in the model training process, it is very likely to improve the training efficiency , while reducing the hardware configuration and improving its intelligence.

Secondly, the self-play link in model training is very important, so is it possible to introduce possible innovations in this link? We think we can start with the bootstrap mechanism of self-play. **In both alpha go and alpha zero, the guidance mechanism uses Monte Carlo tree search (only alpha zero does not use MCTS rollouts in the process of real chess), considering that the search space of chess is obviously smaller than that of Go, we can try Reach deeper depths by customizing the pruning search given to the Monte Carlo tree for better results.**
