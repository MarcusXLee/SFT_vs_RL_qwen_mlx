import yaml
from box import Box
from evaluation.evaluator import evaluator_init
from tqdm import tqdm
import datetime

try:
    import accelerate  # noqa: F401  (imported for side effects on the GPU path)
except ImportError:
    accelerate = None

from utils_general import load_config
from utils_rl import set_str_action_space

def main(config, boxed_config):
    action_space = set_str_action_space(boxed_config.env_config)

    print(yaml.dump(config))
    daytime =  datetime.datetime.now().strftime("%Y-%m-%d||%H:%M:%S")
    player = evaluator_init[boxed_config.evaluator](action_space = action_space, daytime=daytime, **boxed_config)
    if getattr(boxed_config, 'best_of_n_baseline', 0) > 0:
        player.baseline(getattr(boxed_config, 'best_of_n_baseline', 0))
    else:
        player.evaluate()


if __name__ == "__main__":
    config, boxed_config = load_config()
    print(config)
    main(config, boxed_config)
