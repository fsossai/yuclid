from datetime import datetime
import itertools
import json
import pandas as pd
import sys
import subprocess
import re

with open("yuclid.json") as f:
    data = json.load(f)

ndims = len(data["space"])

dims = data["space"].keys()

space = dict()
for key, value in data["space"].items():
    if key.endswith(":py"):
        name = key.split(":")[-2]
        space[name] = eval(value)
    elif key.endswith(".sh"):
        pass
    else:
        space[key] = value

order = list(space.keys())
for key in data.get("order", []):
    order.append(order.pop(order.index(key)))

varmap = {k: i for i, k in enumerate(order)}
space_order = [space[x] for x in order]



columns = list(space.keys()) + list(data["metric"].keys())
df = pd.DataFrame(columns=columns)

def substitute_vars(x, point, point_id):
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\}"
    y = re.sub(pattern, lambda m: str(point[varmap[m.group(1)]]), x)
    pattern = r"\$\{yuclid\.\#\}"
    y = re.sub(pattern, lambda m: point_id, y)
    pattern = r"\$\{yuclid\.([a-zA-Z0-9_]+)\.\*\}"
    y = re.sub(pattern, lambda m: " ".join(space[m.group(1)]), y)
    return y

def run_trials(f):
    for i, point in enumerate(itertools.product(*space_order)):
        point_id = f"yuclid.{i:08x}.tmp"
        command = substitute_vars(data["trial"], point, point_id)
        subprocess.run(command, shell=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        result = dict(zip(order, point))
        for metric, command in data["metric"].items():
            command = substitute_vars(command, point, point_id)
            out = subprocess.run(command, shell=True, text=True,
                                 capture_output=True)
            value = float(out.stdout.strip())
            result[metric] = value
        # df.loc[len(df)] = result
        print(*result.values())
        f.write(json.dumps(result) + "\n")
        f.flush()

output_file = "trials.{:%Y%m%d-%H%M}.json".format(datetime.now())
with open("result.json", "a") as f:
    run_trials(f)
