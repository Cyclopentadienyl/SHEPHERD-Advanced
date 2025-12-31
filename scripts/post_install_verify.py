import json, sys, subprocess, shutil

def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output

report = {"cuda_visible_devices": None, "nvidia_smi": None, "nvcc": None}

report["cuda_visible_devices"] = sys.argv[1] if len(sys.argv) > 1 else None

rc, out = run(["nvidia-smi"])
report["nvidia_smi"] = out.strip() if rc == 0 else None

nvcc = shutil.which("nvcc")
if nvcc:
    rc, out = run([nvcc, "--version"])
    report["nvcc"] = out.strip() if rc == 0 else None

print(json.dumps(report, indent=2))
