
import gym
import numpy as np
from mdp.policyEvaluation import synchronous_state_sweep_policy_evaluation
from mdp.policyEvaluation import asynchronous_inplace_policy_evaluation

def compare_policies(policy_1: dict,
                     policy_2: dict,
                     numberActions: int,
                     numberStates: int,
                     )->bool:
    policy_array_1 = []
    policy_array_2 = []
    for s in range(numberStates):
        for a in range(numberActions):
            policy_array_1.append(policy_1[s][a])
            policy_array_2.append(policy_2[s][a])

    return  policy_array_1 == policy_array_2


def improvePolicy(env,
                  valueFunctionVector: np.ndarray,
                  numberActions: int,
                  numberStates: int,
                  discountRate: float):

    # this matrix will store the q-values (action value functions) for every state
    # this matrix is returned by the function
    qvaluesMatrix = np.zeros((numberStates, numberActions))
    # this is the improved policy
    # this matrix is returned by the function
    improvedPolicy= {}
    for state in range(numberStates):
        # for action in range(numberActions):
        #     improvedPolicy[state] = {action: 0}
        improvedPolicy[state] = {i: 0 for i in range(numberActions)}
    # improvedPolicy = np.zeros((numberStates, numberActions))

    for stateIndex in range(numberStates):
        # computes a row of the qvaluesMatrix[stateIndex,:] for fixed stateIndex,
        # this loop iterates over the actions
        for actionIndex in range(numberActions):
            # computes the Bellman equation for the action value function
            for probability, nextState, reward, isTerminalState in env.P[stateIndex][actionIndex]:
                qvaluesMatrix[stateIndex, actionIndex] = qvaluesMatrix[stateIndex, actionIndex] + probability * (
                            reward + discountRate * valueFunctionVector[nextState])

        # find the action indices that produce the highest values of action value functions
        bestActionIndex = np.where(qvaluesMatrix[stateIndex, :] == np.max(qvaluesMatrix[stateIndex, :]))[0][0]
        improvedPolicy[stateIndex][bestActionIndex] = 1 #/ np.size(bestActionIndex)
    return improvedPolicy, qvaluesMatrix

def policyIteration(env,
                    discountRate : float=0.9,
                   stateNumber:int=16,
                   actionNumber:int=4,
                   maxNumberOfIterationsOfPolicyIteration:int=1000,
                    maxNumberOfIterationsOfIterativePolicyEvaluation:int=1000,
                    convergenceToleranceIterativePolicyEvaluation:float=10**(-6),
                    synchronous_evaluation: bool = True
                   )->dict:

    # initialPolicy=(1/actionNumber)*np.ones((stateNumber,actionNumber))
    initialPolicy= {}
    for state in range(stateNumber):
        initialPolicy[state] =  {i: 1/actionNumber for i in range(actionNumber)}
    # policy = generate_arbitrary_policy(env)
    valueFunctionVectorInitial=np.zeros(env.observation_space.n)

    for iteration in range(maxNumberOfIterationsOfPolicyIteration):
        print("Iteration - {} - of policy iteration algorithm".format(iteration))
        if (iteration == 0):
            currentPolicy=initialPolicy

        if synchronous_evaluation:
            valueFunctionVectorComputed, _ =synchronous_state_sweep_policy_evaluation(env,
                                                                                  currentPolicy,
                                                                                  discountRate,

                                                                                  convergenceToleranceIterativePolicyEvaluation,
                                                                                  maxNumberOfIterationsOfIterativePolicyEvaluation)
        else:
            valueFunctionVectorComputed, _ =asynchronous_inplace_policy_evaluation(env,
                                                                                  currentPolicy,
                                                                                  discountRate,

                                                                                  convergenceToleranceIterativePolicyEvaluation,
                                                                                  maxNumberOfIterationsOfIterativePolicyEvaluation)


        improvedPolicy,qvaluesMatrix=improvePolicy(env,
                                                   valueFunctionVectorComputed,
                                                   actionNumber,
                                                   stateNumber,
                                                   discountRate)
        # if two policies are equal up to a certain "small" tolerance
        # then break the loop - our algorithm converged


        if compare_policies(currentPolicy ,
            improvedPolicy ,
            actionNumber ,
            stateNumber ,
            ):
            currentPolicy=improvedPolicy
            print("Policy iteration algorithm converged!")
            break
        currentPolicy=improvedPolicy
    return currentPolicy