import os
import subprocess
import glob
import time

def run_benchmarks():
    baselines_dir = "../paper_experiments/training/data/paper_baselines/es_finetuned"
    models = glob.glob(f"{baselines_dir}/*.omnibit")
    
    if not models:
        print("No .omnibit models found in es_finetuned!")
        return
        
    print(f"🏁 Found {len(models)} models to benchmark on ESP32 HIL.")
    print("🧹 Cleaning up old results...")
    if os.path.exists("master_hil_summary.csv"):
        os.remove("master_hil_summary.csv")
    
    for i, model in enumerate(models):
        model_name = os.path.basename(model)
        print(f"\n{'='*60}")
        print(f" 🏎️  [{i+1}/{len(models)}] Benchmarking {model_name}")
        print(f"{'='*60}")
        
        # Phase 1: Benchmark (20 Episodes)
        print(f"📊 Phase 1: Running 20 episodes for statistics...")
        cmd_bench = ["../.venv/bin/python", "-u", "run_f110_vegas_hil.py", 
                     "--model", model, 
                     "--episodes", "20", 
                     "--log-csv"]
        
        result = subprocess.run(cmd_bench)
        if result.returncode != 0:
            print(f"⚠️ Warning: Benchmark step had an error for {model_name}")
            
        print("⏳ Waiting 5 seconds for USB buffers to flush...")
        time.sleep(5)
        
        # Phase 2: Trajectory Plot (1 Episode)
        print(f"🗺️  Phase 2: Generating trajectory plot (1 episode)...")
        cmd_traj = ["../.venv/bin/python", "-u", "run_f110_vegas_hil.py", 
                    "--model", model, 
                    "--episodes", "1", 
                    "--render"]
        subprocess.run(cmd_traj)
        
        print("⏳ Waiting 5 seconds before next model...")
        time.sleep(5)
        
    print(f"\n{'='*60}")
    print("✅ All benchmarks completed!")
    print("📈 Generating final comparison plot...")
    
    subprocess.run(["../.venv/bin/python", "generate_hil_plot.py"])
    print("🎉 Done! Check master_hil_summary.csv and the generated PNG plots.")

if __name__ == "__main__":
    run_benchmarks()
