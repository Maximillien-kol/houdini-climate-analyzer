"""
AgriShield AI — Master Launcher
================================
Run everything from one file.

Commands
--------
  python run.py               # install deps → retrain if needed → start both servers
  python run.py retrain       # parse PDFs + train all models
  python run.py serve         # start Python ML service + Node.js API
  python run.py install       # install all Python + Node.js dependencies
  python run.py train         # synthetic-only training (no documents)
"""

import os
import sys
import subprocess
import time

ROOT    = os.path.dirname(os.path.abspath(__file__))
ML_DIR  = os.path.join(ROOT, "python_ml")
API_DIR = os.path.join(ROOT, "nodejs_api")

ARTIFACT = os.path.join(ML_DIR, "artifacts", "rain_prediction_rf.pkl")

# On Windows npm/node are .cmd shell scripts and require shell=True
IS_WIN = sys.platform == "win32"
NPM  = "npm.cmd"  if IS_WIN else "npm"
NODE = "node.exe" if IS_WIN else "node"


# ── helpers ────────────────────────────────────────────────────────────────────

def banner(text: str):
    line = "═" * 54
    print(f"\n╔{line}╗")
    print(f"║  {text:<52}║")
    print(f"╚{line}╝\n")


def step(n: str, text: str):
    print(f"  [{n}] {text}")


def run(cmd: list, cwd: str, check=True):
    """Run a command, stream output live, raise on failure."""
    result = subprocess.run(cmd, cwd=cwd, shell=IS_WIN)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode


def popen(cmd: list, cwd: str) -> subprocess.Popen:
    """Start a long-running background process."""
    return subprocess.Popen(cmd, cwd=cwd, shell=IS_WIN)


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_install():
    banner("Step: Install Dependencies")

    step("1/2", "Python dependencies (pip install -r requirements.txt) …")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        cwd=ML_DIR)
    print("        ✓ Python deps ready")

    step("2/2", "Node.js dependencies (npm install) …")
    env_example = os.path.join(API_DIR, ".env.example")
    env_file    = os.path.join(API_DIR, ".env")
    if not os.path.exists(env_file) and os.path.exists(env_example):
        import shutil
        shutil.copy(env_example, env_file)
        print("        ✓ Copied .env.example → .env")
    run([NPM, "install", "--silent"], cwd=API_DIR)
    print("        ✓ Node deps ready\n")


def cmd_train():
    banner("Step: Train Models  (synthetic only)")
    run([sys.executable, "main.py", "train"], cwd=ML_DIR)


def cmd_retrain():
    banner("Step: Retrain Models  (synthetic + documents)")
    print("  Scanning data/documents/monthly_forecasts/ for PDF/DOCX …\n")
    run([sys.executable, "main.py", "retrain"], cwd=ML_DIR)


def cmd_serve():
    banner("AgriShield AI — Starting Services")

    processes = []

    try:
        # ── Python ML service ─────────────────────────────────────────────────
        step("1/2", "Starting Python ML service on port 5001 …")
        ml_proc = popen([sys.executable, "main.py", "serve"], cwd=ML_DIR)
        processes.append(("Python ML", ml_proc))
        time.sleep(3)

        # ── Node.js API ───────────────────────────────────────────────────────
        step("2/2", "Starting Node.js API gateway on port 3000 …")
        node_cmd = [NODE, "server.js"] if _node_entry_exists() else [NPM, "start"]
        node_proc = popen(node_cmd, cwd=API_DIR)
        processes.append(("Node.js API", node_proc))
        time.sleep(2)

        print()
        print("  ✓ AgriShield AI is running!\n")
        print("     Python ML Service : http://localhost:5001/health")
        print("     Node.js API        : http://localhost:3000/api/health")
        print("     Quick test         : http://localhost:3000/api/data/simulate")
        print()
        print("  Press Ctrl+C to stop both services.\n")

        # Keep alive — wait on both processes
        while True:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"\n[WARN] {name} exited with code {code}. Stopping all.")
                    raise KeyboardInterrupt
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nShutting down …")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"  ✓ {name} stopped")
        print("Done.\n")


def cmd_start():
    """Default: install → retrain if no artifacts → serve."""
    banner("AgriShield AI — Full Start")

    # 1. install dependencies
    cmd_install()

    # 2. train if no artifacts exist
    if not os.path.exists(ARTIFACT):
        print("  No trained models found — running retrain …\n")
        cmd_retrain()
    else:
        print("  ✓ Trained models found. Skipping training.")
        print("    (Run:  python run.py retrain  to retrain with new documents)\n")

    # 3. serve
    cmd_serve()


# ── utils ──────────────────────────────────────────────────────────────────────

def _node_entry_exists() -> bool:
    return os.path.exists(os.path.join(API_DIR, "server.js"))


# ── entry point ────────────────────────────────────────────────────────────────

COMMANDS = {
    "start":   cmd_start,
    "install": cmd_install,
    "train":   cmd_train,
    "retrain": cmd_retrain,
    "serve":   cmd_serve,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"

    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'.")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[cmd]()
