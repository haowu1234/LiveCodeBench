import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    # Support custom run_id suffix
    run_id = getattr(args, 'run_id', None)
    run_id_suffix = f"_{run_id}" if run_id else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}{run_id_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    # Support custom run_id suffix
    run_id = getattr(args, 'run_id', None)
    run_id_suffix = f"_{run_id}" if run_id else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}{run_id_suffix}_eval_all.json"
    return path
