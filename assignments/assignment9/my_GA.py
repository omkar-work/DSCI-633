import numpy as np
import pandas as pd
from pdb import set_trace


class my_GA:
    # Tuning with Genetic Algorithm for model parameters

    def __init__(self, model, data_X, data_y, decision_boundary, obj_func, generation_size=100, selection_rate=0.5,
                 mutation_rate=0.01, crossval_fold=5, max_generation=100, max_life=3):
        # inputs:
        # model: class object of the learner under tuning, e.g. my_DT
        # data_X: training data independent variables (pd.Dataframe)
        # data_y: training data dependent variables (pd.Series or list)
        # decision_boundary: list of boundaries of each decision variable,
        # e.g. decision_boundary = [("gini", "entropy"), [1, 16], [0, 0.1]] for my_DT means:
        # the first argument criterion can be chosen as either "gini" or "entropy"
        # the second argument max_depth can be any number 1 <= max_depth < 16
        # the third argument min_impurity_decrease can be any number 0 <= min_impurity_decrease < 0.1
        # obj_func: generate objectives, all objectives are higher the better
        # generation_size: number of points in each generation
        # selection_rate: percentage of survived points after selection, only affect single objective
        # mutation_rate: probability of being mutated for each decision in each point
        # crossval_fold: number of fold in cross-validation (for evaluation)
        # max_generation: early stopping rule, stop when reached
        # max_life: stopping rule, stop when max_life consecutive generations do not improve
        self.model = model
        self.data_X = data_X
        self.data_y = data_y
        self.decision_boundary = decision_boundary
        self.obj_func = obj_func
        self.generation_size = int(generation_size)
        self.selection_rate = selection_rate  # applies only to singe-objective
        self.mutation_rate = mutation_rate
        self.crossval_fold = int(crossval_fold)
        self.max_generation = int(max_generation)
        self.max_life = int(max_life)
        self.life = self.max_life
        self.iter = 0
        self.generation = []
        self.pf_best = []
        self.evaluated = {None: -1}
        self.lenGen = 0

    def initialize(self):
        # Randomly generate generation_size points to self.generation

        self.generation = []
        for _ in range(self.generation_size):
            x = []
            for decision in self.decision_boundary:
                if type(decision) == list:
                    x.append(np.random.random() * (decision[1] - decision[0]) + decision[0])
                else:
                    x.append(decision[np.random.randint(len(decision))])
            self.generation.append(tuple(x))
        ######################
        # check if size of generation is correct
        self.lenGen = len(self.generation)
        assert (len(self.generation) == self.generation_size)
        # print(self.generation)

        # e.g. decision_boundary = [("gini", "entropy"), [1, 16], [0, 0.1]] for my_DT means:
        # [('entropy', 7.813403091150089, 0.06553293516131108), ('entropy', 4.810365164147403, 0.05480819515903192),
        #  ('entropy', 10.052575908227746, 0.0408416150073975), ('entropy', 4.9880984663380605, 0.0014028201965396892),
        #  ('gini', 9.9733859799947, 0.09717816388900813), ('entropy', 9.35582624741804, 0.0855897495266541),
        #  ('gini', 8.001908643722558, 0.06634948413721495), ('entropy', 6.8608597165276235, 0.005699927426591267),
        #  ('gini', 13.464384113883007, 0.0437738523964795), ('gini', 9.81816391297118, 0.023665363372876703)]
        return self.generation

    def evaluate(self, decision):
        # Evaluate a certain point
        # decision: tuple of decisions
        # Avoid repetitive evaluations
        # print(self.generation)
        # print(decision)
        if decision not in self.evaluated:
            # print(decision)
            # evaluate with self.crossval_fold fold cross-validation on self.data_X and self.data_y
            clf = self.model(*decision)
            # Cross validation:
            indices = [i for i in range(len(self.data_y))]
            np.random.shuffle(indices)
            # print(indices)
            size = int(np.ceil(len(self.data_y) / float(self.crossval_fold)))
            # print("Size : ")
            # print(size)
            objs_crossval = None
            for fold in range(self.crossval_fold):
                start = int(fold * size)
                # print(start)
                end = start + size
                # print(end)
                test_indices = indices[end:]
                # print("Test : "+str(test_indices))
                train_indices = indices[start:end]
                # print("Train : "+str(train_indices))
                X_train = self.data_X.loc[train_indices]
                X_train.index = range(len(X_train))
                X_test = self.data_X.loc[test_indices]
                X_test.index = range(len(X_test))
                y_train = self.data_y.loc[train_indices]
                y_train.index = range(len(y_train))
                y_test = self.data_y.loc[test_indices]
                y_test.index = range(len(y_test))
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                pred_proba = clf.predict_proba(X_test)
                actuals = y_test
                objs = np.array(self.obj_func(predictions, actuals, pred_proba))
                # print(str(objs))
                if type(objs_crossval) == type(None):
                    objs_crossval = objs
                else:
                    objs_crossval += objs

            #output ranges in 0.007 with division and is near to expected without division.
            objs_crossval = objs_crossval #/ float(len(self.data_y))
            self.evaluated[decision] = objs_crossval
            # print("Problem maybe "+str(self.evaluated[decision]))
        return self.evaluated[decision]

    def is_better(self, a, b):
        # Check if decision a binary dominates decision b
        # Return 0 if a == b,
        # Return 1 if a binary dominates b,
        # Return -1 if a does not binary dominates b.
        if a == b:
            return 0
        obj_a = self.evaluate(a)
        obj_b = self.evaluate(b)

        runningTotal = 0
        for count in range(len(obj_a)): #keep track of whether each element is > or < than corresponding element for a and b
            if obj_a[count] > obj_b[count]:
                runningTotal += 1
            elif obj_a[count] < obj_b[count]:
                runningTotal -= 1
            else:
                pass

        if runningTotal > 0:
            return 1
        elif runningTotal == 0:
            return 0
        else:
            return -1
        # write your own code below
        # if (obj_a[i] > obj_b[i] for i in range(len(obj_a))):
        #     return 1
        # else:
        #     return -1

    def compete(self, pf_new, pf_best):
        # Compare and merge two pareto frontiers
        # If one point y in pf_best is binary dominated by another point x in pf_new
        # (exist x and y; self.is_better(x, y) == 1)
        # replace that point y in pf_best with the point x in pf_new
        # If one point x in pf_new is not dominated by any point y in pf_best (and does not exist in pf_best)
        # (forall y in pf_best; self.is_better(y, x) == -1)
        # add that point x to pf_best
        # Return True if pf_best is modified in the process, otherwise return False
        # Write your own code below
        # print("Inside compete")
        # print(self.generation)
        modified = False
        for i in range(len(pf_best)):
            for j in range(len(pf_new)):
                if (self.is_better(pf_best[i], pf_best[j])) == 1:
                    pf_best[i] = pf_new[j]
                    pf_new.pop(j)
                    modified = True
                    break
        to_add = []
        for j in range(len(pf_new)):
            not_dominated = True
            for i in range(len(pf_best)):
                if (self.is_better(pf_new[j], pf_best[i])) == -1:
                    not_dominated = False
                    break
            if not_dominated:
                to_add.append(j)
                modified = True
        for j in to_add:
            pf_best.append(pf_new[j])
        return modified

    def select(self):
        # Select which points will survive based on the objectives
        # Update the following:
        # self.pf = pareto frontier (undominated points from self.generation)
        # self.generation = survived points
        # single-objective:
        # print("Inside select")
        # #print(self.generation)
        # print("This "+str(self.evaluate(self.generation[0])))
        if (self.evaluate(self.generation[0])).size == 1:   #used size because len() was throwing error : TypeError: len() of unsized object
            selected = np.argsort([self.evaluate(x)[0] for x in self.generation])[::-1][
                       :int(np.ceil(self.selection_rate * self.generation_size))]
            self.pf = [self.generation[selected[0]]]
            self.generation = [self.generation[i] for i in selected]
            # print("Here")
            # print(self.generation)
        # multi-objective:
        else:
            self.pf = []
            for x in self.generation:
                if not np.array([self.is_better(y, x) == 1 for y in self.generation]).any():
                    self.pf.append(x)
            # remove duplicates
            self.pf = list(set(self.pf))
            # Add second batch undominated points into next generation if only one point in self.pf
            if len(self.pf) == 1:
                self.generation.remove(self.pf[0])
                next_pf = []
                for x in self.generation:
                    if not np.array([self.is_better(y, x) == 1 for y in self.generation]).any():
                        next_pf.append(x)
                next_pf = list(set(next_pf))
                self.generation = self.pf + next_pf
                # print("this if")
                # print(self.generation)
            else:
                self.generation = self.pf[:]
        return

    def crossover(self):
        # randomly select two points in self.generation
        # and generate a new point
        # repeat until self.generation_size points were generated
        # Write your own code below

        def cross(a, b):
            new_point = []
            for i in range(len(a)):
                randomNum = np.random.randint(0, 2)
                #print(randomNum)
                if randomNum == 0:
                    new_point.append(a[i])
                else:
                    new_point.append(b[i])
            return tuple(new_point)

        to_add = []
        # print("Gen size = "+str(self.generation_size))
        # print("LenGen = "+str(len(self.generation)))
        for _ in range(self.generation_size - len(self.generation)):
            ids = np.random.choice(len(self.generation), 2, replace=False)
            # print("Inside crossover")
            # print(self.generation[ids[0]], self.generation[ids[1]])
            new_point = cross(self.generation[ids[0]], self.generation[ids[1]])
            # print(new_point)
            to_add.append(new_point)
            # self.generation.extend(new_point)
        self.generation.extend(to_add)
        ######################
        # check if size of generation is correct
        assert (len(self.generation) == self.generation_size)
        return self.generation

    def mutate(self):
        # Uniform random mutation:
        # each decision value in each point of self.generation
        # has the same probability self.mutation_rate of being mutated
        # to a random valid value
        # write your own code below

        for i, x in enumerate(self.generation):
            new_x = list(x)
            for j in range(len(x)):
                if np.random.random() < self.mutation_rate:
                    decision = self.decision_boundary[j]
                    if type(decision) == list:
                        new_x[j] = np.random.random() * (decision[1] - decision[0]) + decision[0]
                    else:
                        new_x[j] = decision[np.random.randint(len(decision))]
            self.generation[i] = tuple(new_x)
        return self.generation

    def tune(self):
        # Main function of my_GA Stop when self.iter == self.max_generation or self.life == 0 Return the best pareto
        # frontier pf_best (list of decisions that never get binary dominated by any candidate evaluated)
        self.initialize()
        while self.life > 0 and self.iter < self.max_generation:
            self.select()
            if self.compete(self.pf, self.pf_best):
                self.life = self.max_life
            else:
                self.life -= 1
            self.iter += 1
            self.crossover()
            self.mutate()
        return self.pf_best
