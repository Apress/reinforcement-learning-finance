import numpy as np
from enum import Enum, unique
import logging
import time

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@unique
class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class EpsGreedyPolicy(object):
    def __init__(self, eps):
        self.epsilon = eps
        np.random.seed(50)

    def nextAction(self, state, Q):
        if np.random.random() < self.epsilon:
            return Actions(np.random.choice(4))

        maxval = MazeSolver.NEG_INFTY
        act = None
        for action in Actions:
            if maxval < Q[state[0], state[1], action.value]:
                maxval = Q[state[0], state[1], action.value]
                act = action

        if act is None:
            act = action
        return act


class MazeSolver(object):
    NEG_INFTY = float(-1E10)
    EPSILON = 0.1

    def __init__(self, entry, exit):
        self.gamma = 1.0
        self.maxIter = 5000
        self.mazeSize = (12, 12)
        self.nActions = len(Actions)
        self.entry = entry
        self.exit = exit
        self.walls =      {(5,1), (5,2), (5,3), (5,4), (5,5),
                           (7,3), (7,4), (7,5), (7,6),
                           (2,7), (3,7), (4,7), (5,7), (6,7), (7,7), (8,7),
                           (4,9), (5,9), (6,9), (7,9), (8,9), (9,9), (10,9),
                           (9,4), (10,4)}
        self.actionMap = {Actions.LEFT : (-1, 0),
                          Actions.RIGHT : (1, 0),
                          Actions.UP : (0, -1),
                          Actions.DOWN : (0, 1)}
        # add bounding walls
        for i in range(self.mazeSize[0]):
            self.walls.add((i, 0))
            self.walls.add((i, self.mazeSize[1]-1))

        for j in range(self.mazeSize[1]):
            self.walls.add((0, j))
            self.walls.add((self.mazeSize[0]-1, j))
        if self.entry in self.walls:
            raise ValueError("Entry square is inadmissible")
        if self.exit in self.walls:
            raise ValueError("Exit square is inadmissible")
        self.QStar = np.ndarray((self.mazeSize[0], self.mazeSize[1], self.nActions), dtype=np.float64)
        self.initQStar(self.QStar)
        self.policy = EpsGreedyPolicy(0.1)

    def policyEvaluationAndImp(self):
        # Using a greedy policy with updated Q implies an implicit policy improvement
        itercount = 0
        while itercount < self.maxIter:
            itercount += 1
            for i in range(self.mazeSize[0]):
                for j in range(self.mazeSize[1]):
                    state = (i,j)
                    if state in self.walls:
                        continue
                    action = self.policy.nextAction(state, self.QStar)
                    reward = self.rewardFunc(state, action)
                    nextstate = self.transitionFunc(state, action)
                    if nextstate not in self.walls:
                        nextaction = self.policy.nextAction(nextstate, self.QStar)
                        nextq = self.QStar[nextstate[0], nextstate[1], nextaction.value]
                        newval = reward + self.gamma * nextq
                        self.QStar[state[0], state[1], action.value] = newval
                    else:
                        self.QStar[state[0], state[1], action.value] = reward

    def transitionFunc(self, state0, action):
        increments = self.actionMap[action]
        return state0[0] + increments[0], state0[1] + increments[1]

    def rewardFunc(self, state0, action):
        state1 = self.transitionFunc(state0, action)
        if state1 in self.walls:
            return MazeSolver.NEG_INFTY
        elif state1 == self.exit:
            return 1
        return -1

    def initQStar(self, Q):
        for i in range(self.mazeSize[0]):
            for j in range(self.mazeSize[1]):
                square = (i,j)
                if square in self.walls:
                    for action in Actions:
                        Q[i, j, action.value] = MazeSolver.NEG_INFTY
                else:
                    for action in Actions:
                        Q[i, j, action.value] = 0


    def optPolicy(self):
        # run generalized policy iteration
        self.policyEvaluationAndImp()

        optpath = [self.entry]
        sq = self.entry
        while sq != self.exit:
            bestaction = self.policy.nextAction(sq, self.QStar)
            if bestaction is None:
                return optpath
            sq = self.transitionFunc(sq, bestaction)
            optpath.append(sq)

        return optpath


if __name__ == "__main__":
    entry = (1, 3)
    exit = (1, 12)
    maze_solver = MazeSolver(entry, exit)
    path = maze_solver.optPolicy()
    if path[-1] != exit:
        logger.info("No path exists")

    logger.info("->".join([str(p) for p in path]))


