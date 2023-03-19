import subprocess
import json
import time

seed = 1


def run_control(hidden_size, seed, total_timesteps):
    cmd = ['python', 'ppo.py', '--capture-video', 
           '--hidden-size', str(hidden_size),
           '--seed', str(seed),
           '--total-timesteps', str(total_timesteps)]
    subprocess.run(cmd)

def run_sns(hidden_size, seed, total_timesteps):
    cmd = ['python', 'ppo_sns.py', '--capture-video', 
           '--hidden-size', str(hidden_size),
           '--seed', str(seed),
           '--total-timesteps', str(total_timesteps)]
    subprocess.run(cmd)


if __name__ == "__main__":
    filename = 'experiments.json'

    with open(filename, 'r') as f:
        data = json.load(f)

   # Generate the list of hidden_sizes based on the JSON data
    if "hidden_sizes" in data:
        if "min" in data["hidden_sizes"] and "max" in data["hidden_sizes"] and "num_between" in data["hidden_sizes"]:
            hidden_sizes_min = data['hidden_sizes']['min']
            hidden_sizes_max = data['hidden_sizes']['max']
            hidden_sizes_num_between = data['hidden_sizes']['num_between']
            hidden_sizes = [int(hidden_sizes_min + i*(hidden_sizes_max-hidden_sizes_min)/(hidden_sizes_num_between-1)) for i in range(hidden_sizes_num_between)]
        elif "list" in data["hidden_sizes"]:
            hidden_sizes = data["hidden_sizes"]["list"]
            hidden_sizes = [int(size) for size in hidden_sizes]
        else:
            print("Invalid JSON format for hidden_sizes")
            exit()
    else:
        print("hidden_sizes not found in JSON")
        exit()
    
    # Generate the list of total_timesteps based on the JSON data
    if "total_timesteps" in data:
        if "min" in data["total_timesteps"] and "max" in data["total_timesteps"] and "num_between" in data["total_timesteps"]:
            total_timesteps_min = data['total_timesteps']['min']
            total_timesteps_max = data['total_timesteps']['max']
            total_timesteps_num_between = data['total_timesteps']['num_between']
            total_timesteps = [int(total_timesteps_min + i*(total_timesteps_max-total_timesteps_min)/(total_timesteps_num_between-1)) for i in range(total_timesteps_num_between)]
        elif "list" in data["total_timesteps"]:
            total_timesteps = data["total_timesteps"]["list"]
            total_timesteps = [int(step) for step in total_timesteps]
        else:
            print("Invalid JSON format for total_timesteps")
            exit()
    else:
        print("total_timesteps not found in JSON")
        exit()
    
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Total timesteps: {total_timesteps}")     

    for size in hidden_sizes:
        for timesteps in total_timesteps:
            run_control(size,seed,timesteps)
            run_sns(size, seed, timesteps)
            

    json_dict = {
        "hidden_sizes": {
            "list": hidden_sizes
        },
        "total_timesteps": {
            "list": total_timesteps
        }
    }
    
    # Get current time as timestamp string
    timestamp = str(int(time.time()))
    
    # Write JSON to file
    filename = f"run_stats/experiments_{timestamp}.json"
    with open(filename, "w") as json_file:
        json.dump(json_dict, json_file)

