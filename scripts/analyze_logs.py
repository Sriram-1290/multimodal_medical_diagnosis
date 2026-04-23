import os
from tensorboard.backend.event_processing import event_accumulator

def analyze_logs():
    log_dir = "models/logs"
    if not os.path.exists(log_dir):
        print("Log directory not found.")
        return

    # Find the most recent event file
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print("No event files found.")
        return

    # Sort by timestamp (filename contains it)
    event_files.sort(key=os.path.getmtime)
    
    print(f"Analysis of Training Logs (Total runs: {len(event_files)})\n")
    print(f"{'Run Date':<25} | {'Final Loss':<12}")
    print("-" * 45)

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            if 'Loss/train' in ea.Tags()['scalars']:
                loss_events = ea.Scalars('Loss/train')
                if loss_events:
                    mtime = os.path.getmtime(event_file)
                    from datetime import datetime
                    dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    final_loss = loss_events[-1].value
                    print(f"{dt:<25} | {final_loss:<12.4f}")
        except Exception:
            continue

if __name__ == "__main__":
    analyze_logs()
