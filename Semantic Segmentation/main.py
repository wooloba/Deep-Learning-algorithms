# WARNING: DO NOT CHANGE THIS FILE IN ANYWAY!!!!

import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat


def compute_score(acc, min_thres, max_thres, weight):
  if acc <= min_thres:
    base_score = 0.0
  elif acc >= max_thres:
    base_score = 100.0
  else:
    base_score = float(acc - min_thres) / (max_thres - min_thres) * 100
  return base_score * weight


def run(algorithm, algorithm_name='Algorithm'):
  print('Running {}...'.format(algorithm_name))
  start = timeit.default_timer()
  np.random.seed(0)
  accuracy = algorithm.run()
  np.random.seed()
  stop = timeit.default_timer()
  run_time = stop - start

  print("Result for {}: ".format(algorithm_name))
  print("Accuracy: {:5f} \tTime: {:2f}".format(accuracy, run_time))
  return accuracy, run_time


if __name__ == '__main__':
  result = [
    OrderedDict(
      first_name='Insert your First name here',
      last_name='Insert your Last name here',
    )
  ]

  algorithms = []
  try:
    import semantic_segmentation
    algorithms.append(
      dict(
        algorithm=semantic_segmentation,
        min_thres=0.875,
        max_thres=0.975,
        weight=1,
      ))
  except:
    pass

  total_score = 0.0
  for algorithm in algorithms:
    try:
      accuracy, run_time = run(algorithm['algorithm'], algorithm_name=algorithm['algorithm'].__name__)
      score = compute_score(accuracy, algorithm['min_thres'], algorithm['max_thres'], algorithm['weight'])
      result.append(OrderedDict(
        algorithm_name=algorithm['algorithm'].__name__,
        accuracy=accuracy,
        score=score,
        run_time=run_time,
      ))
    except BaseException as e:
      score = 0.0
      result.append(OrderedDict(
        algorithm_name=algorithm['algorithm'].__name__,
        error_message=str(e),
        score=score,
      ))
    total_score += score
    with open('result.txt', 'w') as f:
      f.writelines(pformat(result, indent=4))
  result.append(dict(
    total_score=total_score,
  ))
  with open('result.txt', 'w') as f:
    f.writelines(pformat(result, indent=4))
