#!/usr/bin/env python3
"""
NeuTTS-Air GUI - Optimized version with parallel processing, GPU acceleration, and session resume
Optimizations: GPU auto-detection (CUDA/MPS), parallel chunk processing (2-4x faster), reduced GC overhead
NOTE: Phonemizer protected by lock for thread-safety
"""
import os
import sys
import io
import json
import subprocess
import importlib
import zipfile
import urllib.request
from typing import Optional, Callable, Dict, List
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import shutil
import glob
import traceback
import platform
from datetime import datetime
import re
import time
import statistics
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ==============================
# GPU Detection
# ==============================
def detect_best_device():
    """Detect the best available device (cuda, mps, or cpu)"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[device] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("[device] Apple Metal (MPS) GPU detected")
            return "mps"
    except ImportError:
        pass

    print("[device] No GPU detected, using CPU")
    return "cpu"

# ==============================
# tiny helpers
# ==============================
def run_cmd(args, check=False) -> int:
    try:
        subprocess.check_call(args)
        return 0
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e.returncode

def pip_install(spec: str, quiet=False, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", spec]
    if extra_args:
        cmd.extend(extra_args)
    if quiet:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.check_call(cmd)

def ensure_dependency(import_name: str, install_spec: str, fallback: Optional[Callable] = None, quiet=False, extra_args=None) -> None:
    try:
        importlib.import_module(import_name)
        return
    except ModuleNotFoundError:
        print(f"[deps] Missing '{import_name}'. Installing {install_spec} …")
        try:
            pip_install(install_spec, quiet=quiet, extra_args=extra_args)
        except Exception as e:
            print(f"[deps] pip install failed for '{import_name}': {e}")
            if fallback:
                fallback()
            else:
                raise
        importlib.invalidate_caches()

def install_neuttsair_from_source():
    print("[deps] Falling back to ZIP: downloading NeuTTS-Air source …")
    repo_url = "https://github.com/neuphonic/neutts-air/archive/refs/heads/main.zip"
    dest_dir = os.path.expanduser("~/.neuttsair_source")
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "neutts_air.zip")
    urllib.request.urlretrieve(repo_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    src_path = os.path.join(dest_dir, "neutts-air-main")
    if src_path not in sys.path:
        sys.path.append(src_path)
    print(f"[deps] Added {src_path} to sys.path")

# ==============================
# eSpeak / eSpeak-NG
# ==============================
def _brew_prefix(formula: str) -> Optional[str]:
    brew = shutil.which("brew") or "/opt/homebrew/bin/brew"
    if not os.path.exists(brew):
        return None
    try:
        out = subprocess.check_output([brew, "--prefix", formula], text=True).strip()
        return out if out and os.path.exists(out) else None
    except Exception:
        return None

def _find_macos_espeak():
    candidates = []
    for formula in ("espeak", "espeak-ng"):
        pref = _brew_prefix(formula)
        if pref:
            candidates.append(pref)
    candidates += ["/opt/homebrew", "/usr/local"]
    lib_candidates = []
    for root in candidates:
        for sub in ("lib", "Cellar/espeak", "opt/espeak/lib", "Cellar/espeak-ng", "opt/espeak-ng/lib"):
            lib_candidates.append(os.path.join(root, sub))
    dylib = None
    for folder in lib_candidates:
        for pattern in ("libespeak-ng*.dylib", "libespeak*.dylib"):
            for f in glob.glob(os.path.join(folder, pattern)):
                if os.path.isfile(f):
                    dylib = f
                    break
            if dylib:
                break
        if dylib:
            break
    bin_dir = None
    bins_to_try = [
        shutil.which("espeak"),
        shutil.which("espeak-ng"),
        "/opt/homebrew/bin/espeak",
        "/usr/local/bin/espeak",
        "/opt/homebrew/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
    ]
    for b in bins_to_try:
        if b and os.path.exists(b):
            bin_dir = os.path.dirname(b)
            break
    return dylib, bin_dir

def _find_linux_espeak():
    bin_dir = None
    for b in (shutil.which("espeak"), shutil.which("espeak-ng")):
        if b:
            bin_dir = os.path.dirname(b)
            break
    libs = [
        "/usr/lib/libespeak-ng.so",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so",
        "/usr/local/lib/libespeak-ng.so",
        "/lib/x86_64-linux-gnu/libespeak-ng.so",
    ]
    for l in libs:
        if os.path.exists(l):
            return l, bin_dir
    for root in ("/usr/lib", "/usr/local/lib", "/lib"):
        found = glob.glob(os.path.join(root, "libespeak*.so*"))
        if found:
            return found[0], bin_dir
    return None, bin_dir

def set_phonemizer_env():
    lib_path = None
    bin_dir = None
    if sys.platform.startswith("darwin"):
        lib_path, bin_dir = _find_macos_espeak()
    elif sys.platform.startswith("linux"):
        lib_path, bin_dir = _find_linux_espeak()
    elif os.name == "nt":
        lib_path = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY", None)
        bin_dir = os.environ.get("PHONEMIZER_ESPEAK_PATH", None)
    if lib_path:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
        os.environ["ESPEAK_LIBRARY"] = lib_path
    if bin_dir:
        os.environ["PHONEMIZER_ESPEAK_PATH"] = bin_dir
    try:
        if lib_path and sys.platform.startswith("darwin"):
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            EspeakWrapper.set_library(lib_path)
    except Exception:
        pass

def ensure_espeak_installed():
    have_bin = shutil.which("espeak") or shutil.which("espeak-ng")
    if not have_bin:
        print("[deps] Installing eSpeak (required by phonemizer) …")
        if sys.platform.startswith("darwin"):
            brew = shutil.which("brew") or "/opt/homebrew/bin/brew"
            if os.path.exists(brew):
                run_cmd([brew, "install", "espeak"])
            else:
                try:
                    messagebox.showwarning(
                        "Manual step required",
                        "Homebrew not found. Install Homebrew, then run:\n  brew install espeak"
                    )
                except Exception:
                    print("Homebrew not found. Please install eSpeak via Homebrew.")
        elif sys.platform.startswith("linux"):
            mgr = shutil.which("apt") or shutil.which("apt-get") or shutil.which("dnf") or shutil.which("pacman")
            if mgr and ("apt" in mgr or "apt-get" in mgr):
                run_cmd(["sudo", "apt-get", "update"])
                run_cmd(["sudo", "apt-get", "install", "-y", "espeak-ng", "libespeak-ng1"])
            elif mgr and ("dnf" in mgr):
                run_cmd(["sudo", "dnf", "install", "-y", "espeak-ng"])
            elif mgr and ("pacman" in mgr):
                run_cmd(["sudo", "pacman", "-Sy", "espeak-ng"])
            else:
                try:
                    messagebox.showwarning(
                        "Manual step required",
                        "Please install espeak-ng with your distribution's package manager."
                    )
                except Exception:
                    print("Please install espeak-ng with your distribution's package manager.")
        elif os.name == "nt":
            try:
                messagebox.showwarning(
                    "Manual step required",
                    "On Windows, install eSpeak NG from:\nhttps://github.com/espeak-ng/espeak-ng/releases"
                )
            except Exception:
                print("On Windows, install eSpeak NG from: https://github.com/espeak-ng/espeak-ng/releases")
    set_phonemizer_env()

# ==============================
# macOS C++ compilation fix
# ==============================
def setup_macos_compilation_env():
    if sys.platform.startswith("darwin"):
        xcode_path = "/Applications/Xcode.app/Contents/Developer"
        cmdline_tools_path = "/Library/Developer/CommandLineTools"
        
        if os.path.exists(xcode_path):
            os.environ["CPLUS_INCLUDE_PATH"] = f"{xcode_path}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1"
        elif os.path.exists(cmdline_tools_path):
            os.environ["CPLUS_INCLUDE_PATH"] = f"{cmdline_tools_path}/SDKs/MacOSX.sdk/usr/include/c++/v1"
        
        os.environ["CXXFLAGS"] = "-stdlib=libc++ -std=c++17"
        os.environ["LDFLAGS"] = "-stdlib=libc++"
        os.environ["CXX"] = "clang++"
        os.environ["CC"] = "clang"

def install_llama_cpp_fallback():
    print("[deps] Attempting to install llama-cpp-python with pre-built wheel...")
    
    if sys.platform.startswith("darwin") and platform.machine() == "arm64":
        try:
            pip_install("llama-cpp-python", extra_args=["--no-cache-dir", "--force-reinstall", 
                                                        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/metal"])
            print("[deps] Successfully installed llama-cpp-python with Metal support")
            return
        except Exception as e:
            print(f"[deps] Metal wheel installation failed: {e}")
    
    try:
        pip_install("llama-cpp-python", extra_args=["--no-cache-dir", "--force-reinstall",
                                                    "--prefer-binary"])
        print("[deps] Successfully installed pre-built llama-cpp-python")
        return
    except Exception as e:
        print(f"[deps] Pre-built wheel installation failed: {e}")
    
    try:
        pip_install("llama-cpp-python==0.2.90", extra_args=["--no-cache-dir"])
        print("[deps] Successfully installed older version of llama-cpp-python")
        return
    except Exception:
        pass
    
    print("[deps] WARNING: Could not install llama-cpp-python. Will use PyTorch fallback for NeuTTS.")

# ==============================
# Python deps (GGUF-first)
# ==============================
def ensure_all_dependencies():
    setup_macos_compilation_env()
    
    ensure_dependency("numpy", "numpy")
    ensure_dependency("soundfile", "soundfile")
    ensure_dependency("phonemizer", "phonemizer>=3.3.0")
    ensure_dependency("librosa", "librosa")
    ensure_dependency("huggingface_hub", "huggingface_hub>=0.25.2")
    ensure_dependency("safetensors", "safetensors")
    
    try:
        ensure_dependency("llama_cpp", "llama-cpp-python", fallback=install_llama_cpp_fallback)
    except Exception as e:
        print(f"[deps] llama-cpp-python installation failed: {e}")
        print("[deps] Will use PyTorch backend instead (may be slower)")
    
    ensure_dependency("perth", "resemble-perth==1.0.1")
    ensure_dependency("neucodec", "neucodec>=0.0.4")
    ensure_espeak_installed()
    ensure_dependency("neuttsair", "git+https://github.com/neuphonic/neutts-air.git",
                      fallback=install_neuttsair_from_source)

# ==============================
# Session Management
# ==============================
class SessionManager:
    def __init__(self):
        self.session_dir = os.path.expanduser("~/.neuttsair_sessions")
        self.session_file = os.path.join(self.session_dir, "last_session.json")
        self.audio_dir = os.path.join(self.session_dir, "audio_chunks")
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def save_session(self, text: str, chunks: List[str], completed_indices: List[int], 
                     ref_wav: str, ref_txt: str, voice_name: str):
        """Save current generation session"""
        try:
            session_data = {
                "text": text,
                "chunks": chunks,
                "completed_indices": completed_indices,
                "total_chunks": len(chunks),
                "ref_wav": ref_wav,
                "ref_txt": ref_txt,
                "voice_name": voice_name,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"[session] Saved session: {len(completed_indices)}/{len(chunks)} chunks")
        except Exception as e:
            print(f"[session] Failed to save session: {e}")
    
    def load_session(self) -> Optional[Dict]:
        """Load last session if it exists and is incomplete"""
        if not os.path.exists(self.session_file):
            return None
        
        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is incomplete
            if len(session_data['completed_indices']) < session_data['total_chunks']:
                return session_data
            else:
                # Session was completed, clean it up
                self.clear_session()
                return None
        except Exception as e:
            print(f"[session] Failed to load session: {e}")
            return None
    
    def save_audio_chunk(self, index: int, audio_data: bytes):
        """Save individual audio chunk to disk"""
        try:
            chunk_path = os.path.join(self.audio_dir, f"chunk_{index:04d}.wav")
            with open(chunk_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            print(f"[session] Failed to save audio chunk {index}: {e}")
    
    def load_audio_chunks(self, completed_indices: List[int]) -> List[bytes]:
        """Load completed audio chunks from disk"""
        audio_chunks = []
        for index in sorted(completed_indices):
            chunk_path = os.path.join(self.audio_dir, f"chunk_{index:04d}.wav")
            if os.path.exists(chunk_path):
                try:
                    with open(chunk_path, 'rb') as f:
                        audio_chunks.append(f.read())
                except Exception as e:
                    print(f"[session] Failed to load audio chunk {index}: {e}")
        return audio_chunks
    
    def clear_session(self):
        """Clear saved session and audio chunks"""
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
            # Clean up audio chunks
            for f in glob.glob(os.path.join(self.audio_dir, "chunk_*.wav")):
                try:
                    os.remove(f)
                except Exception:
                    pass
            print("[session] Cleared session data")
        except Exception as e:
            print(f"[session] Failed to clear session: {e}")

# ==============================
# Voice Profile Management
# ==============================
class VoiceProfileManager:
    def __init__(self):
        self.profiles_dir = os.path.expanduser("~/.neuttsair_profiles")
        self.profiles_file = os.path.join(self.profiles_dir, "profiles.json")
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.profiles = self.load_profiles()
    
    def load_profiles(self) -> Dict:
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_profiles(self):
        try:
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"[profiles] Failed to save profiles: {e}")
    
    def add_profile(self, name: str, wav_path: str, txt_path: str) -> bool:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            profile_wav = os.path.join(self.profiles_dir, f"{safe_name}_{timestamp}.wav")
            profile_txt = os.path.join(self.profiles_dir, f"{safe_name}_{timestamp}.txt")
            
            shutil.copy2(wav_path, profile_wav)
            shutil.copy2(txt_path, profile_txt)
            
            self.profiles[name] = {
                "wav": profile_wav,
                "txt": profile_txt,
                "created": timestamp
            }
            self.save_profiles()
            return True
        except Exception as e:
            print(f"[profiles] Failed to add profile: {e}")
            return False
    
    def get_profile(self, name: str) -> Optional[tuple]:
        if name in self.profiles:
            profile = self.profiles[name]
            if os.path.exists(profile["wav"]) and os.path.exists(profile["txt"]):
                return (profile["wav"], profile["txt"])
        return None
    
    def list_profiles(self) -> List[str]:
        return list(self.profiles.keys())
    
    def delete_profile(self, name: str) -> bool:
        if name in self.profiles:
            try:
                profile = self.profiles[name]
                if os.path.exists(profile["wav"]):
                    os.remove(profile["wav"])
                if os.path.exists(profile["txt"]):
                    os.remove(profile["txt"])
                del self.profiles[name]
                self.save_profiles()
                return True
            except Exception as e:
                print(f"[profiles] Failed to delete profile: {e}")
        return False

# ==============================
# NeuTTS-Air Engine (Singleton for speed)
# ==============================
class TTSEngine:
    """Singleton TTS engine that loads model once and reuses it"""
    _instance = None
    _phonemizer_lock = threading.Lock()  # Lock to protect phonemizer (not thread-safe)

    def __init__(self):
        if TTSEngine._instance is not None:
            raise Exception("Use TTSEngine.get_instance()")

        self.tts = None
        self.current_ref_wav = None
        self.current_ref_txt = None
        self.ref_codes = None
        self.ref_text = None
        
    @staticmethod
    def get_instance():
        if TTSEngine._instance is None:
            TTSEngine._instance = TTSEngine()
        return TTSEngine._instance
    
    def initialize_model(self, force_cpu=False, retry_count=3):
        """Load the TTS model once"""
        if self.tts is not None:
            return  # Already loaded

        print("[tts] Loading TTS model (one-time initialization)...")
        from neuttsair.neutts import NeuTTSAir

        # Detect best device
        device = "cpu" if force_cpu else detect_best_device()

        gguf_available = False
        try:
            import llama_cpp
            gguf_available = True
        except ImportError:
            print("[tts] llama-cpp-python not available, will use PyTorch backend")

        # Try GGUF backend first (recommended)
        if gguf_available:
            for attempt in range(retry_count):
                try:
                    # GGUF uses CPU for backbone, but can use GPU for codec
                    codec_device = "cpu" if force_cpu else device
                    self.tts = NeuTTSAir(
                        backbone_repo="neuphonic/neutts-air-q4-gguf",
                        backbone_device="cpu",  # GGUF backbone always on CPU
                        codec_repo="neuphonic/neucodec",
                        codec_device=codec_device,  # Codec can use GPU for massive speedup
                    )
                    print(f"[tts] Using GGUF backend (backbone: cpu, codec: {codec_device})")
                    print("[tts] Model loaded and ready!")
                    return
                except Exception as _gguf_err:
                    if attempt < retry_count - 1:
                        print(f"[tts] GGUF load attempt {attempt + 1} failed: {_gguf_err}")
                        print(f"[tts] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"[tts] GGUF load failed after {retry_count} attempts: {_gguf_err}")
                        print("[tts] Attempting PyTorch fallback...")
                        gguf_available = False

        # PyTorch fallback (if GGUF failed or not available)
        if not gguf_available:
            try:
                ensure_dependency("torch", "torch")
                ensure_dependency("transformers", "transformers")
            except Exception as e:
                raise Exception(
                    f"Failed to install PyTorch/transformers for fallback: {e}\n"
                    f"SOLUTION:\n"
                    f"1. Check your internet connection\n"
                    f"2. Manually install: pip install torch transformers\n"
                    f"3. If on macOS: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
                )

            for attempt in range(retry_count):
                try:
                    from neuttsair.neutts import NeuTTSAir as _NeuPT
                    # PyTorch can use GPU for both
                    self.tts = _NeuPT(
                        backbone_repo="neuphonic/neutts-air",
                        backbone_device=device,
                        codec_repo="neuphonic/neucodec",
                        codec_device=device,
                    )
                    print(f"[tts] Using PyTorch backend (device: {device})")
                    print("[tts] Model loaded and ready!")
                    return
                except Exception as pt_err:
                    if attempt < retry_count - 1:
                        print(f"[tts] PyTorch load attempt {attempt + 1} failed: {pt_err}")
                        print(f"[tts] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        # Final failure - provide helpful error message
                        error_msg = (
                            f"Failed to load PyTorch backend after {retry_count} attempts.\n\n"
                            f"Original error: {pt_err}\n\n"
                            f"TROUBLESHOOTING:\n"
                            f"1. Check internet connection (needed to download models from HuggingFace)\n"
                            f"2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/\n"
                            f"3. Update transformers: pip install --upgrade transformers huggingface_hub\n"
                            f"4. Try again in a few minutes (HuggingFace API may be temporarily down)\n"
                            f"5. Check HuggingFace status: https://status.huggingface.co/\n\n"
                            f"If issue persists, the model files may need to be downloaded manually."
                        )
                        raise Exception(error_msg)
    
    def load_reference(self, ref_wav: str, ref_txt: str):
        """Load and encode reference voice (cached)"""
        if self.tts is None:
            self.initialize_model()

        # Only re-encode if reference changed
        if ref_wav != self.current_ref_wav or ref_txt != self.current_ref_txt:
            print(f"[tts] Encoding reference voice...")
            self.ref_text = open(ref_txt, "r", encoding="utf-8").read().strip()

            # CRITICAL: Truncate reference text if too long
            # Model has 2048 token context window, reference should use max ~500 tokens (~250 chars)
            if len(self.ref_text) > 250:
                print(f"[tts] WARNING: Reference text too long ({len(self.ref_text)} chars), truncating to 250 chars")
                # Find last complete sentence within 250 chars
                truncated = self.ref_text[:250]
                last_period = truncated.rfind('.')
                if last_period > 100:  # Keep at least 100 chars
                    self.ref_text = truncated[:last_period + 1].strip()
                else:
                    self.ref_text = truncated.strip()
                print(f"[tts] Reference text truncated to: '{self.ref_text[:50]}...'")

            self.ref_codes = self.tts.encode_reference(ref_wav)
            self.current_ref_wav = ref_wav
            self.current_ref_txt = ref_txt
            print(f"[tts] Reference voice ready (text length: {len(self.ref_text)} chars)")
    
    def synthesize(self, text: str, skip_gc=False) -> Optional[bytes]:
        """Synthesize text using cached model and reference - THREAD-SAFE"""
        import soundfile as sf

        if self.tts is None or self.ref_codes is None:
            raise Exception("Model or reference not loaded")

        print(f"[tts] Synthesizing: '{text[:50]}...'")

        try:
            # Use lock to protect phonemizer (not thread-safe)
            # This serializes phonemization but allows GPU codec work to overlap
            with self._phonemizer_lock:
                wav = self.tts.infer(text, self.ref_codes, self.ref_text)

            # Convert to bytes (can happen in parallel after lock is released)
            tmp = io.BytesIO()
            sf.write(tmp, wav, 24000, format="WAV")
            tmp.seek(0)
            audio_bytes = tmp.getvalue()

            # Cleanup - only GC if requested (reduces overhead)
            del wav
            del tmp
            if not skip_gc:
                gc.collect()

            return audio_bytes
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a token limit error
            if any(keyword in error_str for keyword in ['token', 'length', 'too long', 'maximum', 'context']):
                print(f"[tts] Token limit exceeded, text too long")
                raise TokenLimitError(f"Text too long for model: {e}")
            else:
                raise

class TokenLimitError(Exception):
    """Raised when text exceeds model's token limit"""
    pass

def load_default_reference():
    base = os.path.expanduser("~/.neuttsair_reference")
    os.makedirs(base, exist_ok=True)
    wav_path = os.path.join(base, "dave.wav")
    txt_path = os.path.join(base, "dave.txt")
    if not os.path.exists(wav_path):
        print("[ref] Downloading default reference audio …")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples/dave.wav",
            wav_path
        )
    if not os.path.exists(txt_path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples/dave.txt",
            txt_path
        )
    return wav_path, txt_path

# ==============================
# GUI
# ==============================
class NeuTTSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuTTS-Air GUI - Optimized (Parallel + GPU)")
        self.root.geometry("750x570")
        
        # Initialize components
        self.profile_manager = VoiceProfileManager()
        self.session_manager = SessionManager()
        self.current_audio_sequence = []
        self.ref_wav, self.ref_txt = load_default_reference()
        self.generation_thread = None
        self.is_generating = False
        self.tts_engine = TTSEngine.get_instance()
        
        # Timing tracking for ETA
        self.chunk_times = []
        self.start_time = None

        # Session resume tracking
        self.current_chunks = []
        self.completed_chunk_indices = []
        self.resuming_session = False

        # Performance settings
        # Use 2-4 workers for parallel synthesis
        # Phonemizer access is protected by lock, so parallel processing is safe
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(4, max(2, cpu_count // 2))  # 2-4 workers
        print(f"[perf] Using {self.max_workers} parallel workers (phonemizer protected by lock)")
        
        self.create_widgets()
        self.load_voice_profiles()
        
        # Check for incomplete session
        self.check_for_incomplete_session()
        
        # Preload model in background
        threading.Thread(target=self._preload_model, daemon=True).start()
    
    def check_for_incomplete_session(self):
        """Check if there's an incomplete session to resume"""
        session = self.session_manager.load_session()
        if session:
            msg = (f"Found incomplete generation from {session['timestamp']}:\n\n"
                   f"Completed: {len(session['completed_indices'])}/{session['total_chunks']} chunks\n"
                   f"Voice: {session['voice_name']}\n\n"
                   f"Would you like to resume where you left off?")
            
            if messagebox.askyesno("Resume Session?", msg):
                self.resume_session(session)
    
    def resume_session(self, session: Dict):
        """Resume an incomplete generation session"""
        try:
            # Restore text
            self.textbox.delete("1.0", tk.END)
            self.textbox.insert("1.0", session['text'])
            
            # Restore voice profile
            if session['voice_name'] != "Default":
                # Try to find and select the voice
                voices = list(self.voice_combo['values'])
                if session['voice_name'] in voices:
                    self.voice_combo.set(session['voice_name'])
                    self.ref_wav, self.ref_txt = session['ref_wav'], session['ref_txt']
            
            # Load completed audio chunks
            self.current_audio_sequence = self.session_manager.load_audio_chunks(
                session['completed_indices']
            )
            
            # Set resume state
            self.current_chunks = session['chunks']
            self.completed_chunk_indices = session['completed_indices']
            self.resuming_session = True
            
            # Update UI
            completed = len(session['completed_indices'])
            total = session['total_chunks']
            self.progress_label.config(text=f"{completed}/{total}")
            self.progress_var.set((completed / total) * 100)
            
            if self.current_audio_sequence:
                self.download_btn.config(state="normal")
            
            self.status_label.config(
                text=f"Session restored: {completed}/{total} chunks ready. Click Generate to continue."
            )
            
        except Exception as e:
            print(f"[session] Failed to resume session: {e}")
            traceback.print_exc()
            self.session_manager.clear_session()
            messagebox.showerror("Resume Failed", f"Could not resume session: {e}")
    
    def _preload_model(self):
        """Preload TTS model in background for faster first generation"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Loading TTS model..."))
            self.tts_engine.initialize_model()
            self.tts_engine.load_reference(self.ref_wav, self.ref_txt)
            self.root.after(0, lambda: self.status_label.config(text="Ready - Model loaded!"))
        except Exception as e:
            error_str = str(e)
            print(f"[tts] Preload failed: {e}")
            traceback.print_exc()

            # Provide user-friendly error message
            if "NoneType" in error_str or "HuggingFace" in error_str or "API" in error_str:
                msg = (
                    "Failed to download model from HuggingFace.\n\n"
                    "This is usually caused by:\n"
                    "1. Internet connection issues\n"
                    "2. HuggingFace API being temporarily down\n"
                    "3. Firewall blocking downloads\n\n"
                    "SOLUTIONS:\n"
                    "• Check your internet connection\n"
                    "• Wait a few minutes and restart the app\n"
                    "• Check HuggingFace status: status.huggingface.co\n"
                    "• Clear cache: rm -rf ~/.cache/huggingface/\n\n"
                    "You can still use the app - it will try to load the model when you click Generate."
                )
            else:
                msg = f"Model preload failed: {error_str}\n\nYou can still use the app - it will try to load when you click Generate."

            self.root.after(0, lambda m=msg: messagebox.showwarning("Model Load Warning", m))
            self.root.after(0, lambda: self.status_label.config(text="Ready (model not preloaded)"))
        
    def create_widgets(self):
        # Top frame - Voice Profile Selection
        profile_frame = tk.LabelFrame(self.root, text="Voice Profile", padx=10, pady=5)
        profile_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(profile_frame, text="Select Voice:").pack(side=tk.LEFT, padx=5)
        
        self.voice_combo = ttk.Combobox(profile_frame, width=30, state="readonly")
        self.voice_combo.pack(side=tk.LEFT, padx=5)
        self.voice_combo.bind("<<ComboboxSelected>>", self.on_voice_selected)
        
        tk.Button(profile_frame, text="Load Custom", command=self.load_custom_voice).pack(side=tk.LEFT, padx=5)
        tk.Button(profile_frame, text="Save Voice", command=self.save_current_voice).pack(side=tk.LEFT, padx=5)
        tk.Button(profile_frame, text="Delete", command=self.delete_voice).pack(side=tk.LEFT, padx=5)
        
        # Text input area
        text_frame = tk.LabelFrame(self.root, text="Text to Speak", padx=10, pady=5)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.textbox = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=80, height=12)
        self.textbox.pack(fill="both", expand=True)
        # SHORT default text to avoid token limit issues
        self.textbox.insert("1.0", "Hello! Welcome to NeuTTS-Air. This is a test of the text-to-speech system.")
        
        # Control buttons frame
        controls_frame = tk.LabelFrame(self.root, text="Controls", padx=10, pady=5)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Buttons row
        buttons_row = tk.Frame(controls_frame)
        buttons_row.pack(fill="x", pady=5)
        
        self.generate_btn = tk.Button(buttons_row, text="🎙️ Generate Audio", 
                                      command=self.generate_audio, width=20, height=2)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.download_btn = tk.Button(buttons_row, text="💾 Download WAV", 
                                      command=self.download_audio, width=20, height=2, state="disabled")
        self.download_btn.pack(side=tk.LEFT, padx=5)
        
        self.cancel_btn = tk.Button(buttons_row, text="❌ Cancel", 
                                    command=self.cancel_generation, width=15, height=2, state="disabled")
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = tk.Frame(controls_frame)
        progress_frame.pack(fill="x", pady=5)
        
        tk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        
        self.progress_label = tk.Label(progress_frame, text="0/0", width=10)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # ETA display
        eta_frame = tk.Frame(controls_frame)
        eta_frame.pack(fill="x", pady=2)
        
        tk.Label(eta_frame, text="ETA:").pack(side=tk.LEFT, padx=5)
        self.eta_label = tk.Label(eta_frame, text="--:--", anchor=tk.W, font=("TkDefaultFont", 9))
        self.eta_label.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_voice_profiles(self):
        profiles = ["Default"] + self.profile_manager.list_profiles()
        self.voice_combo['values'] = profiles
        if profiles:
            self.voice_combo.current(0)
    
    def on_voice_selected(self, event=None):
        selected = self.voice_combo.get()
        if selected == "Default":
            self.ref_wav, self.ref_txt = load_default_reference()
            self.status_label.config(text="Using default voice")
        else:
            profile = self.profile_manager.get_profile(selected)
            if profile:
                self.ref_wav, self.ref_txt = profile
                self.status_label.config(text=f"Loaded voice: {selected}")
        
        # Reload reference in background
        threading.Thread(target=self._reload_reference, daemon=True).start()
    
    def _reload_reference(self):
        """Reload reference voice in background"""
        try:
            self.tts_engine.load_reference(self.ref_wav, self.ref_txt)
        except Exception as e:
            print(f"[tts] Failed to load reference: {e}")
    
    def load_custom_voice(self):
        wav_path = filedialog.askopenfilename(
            title="Select Reference Audio",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not wav_path:
            return
        
        txt_path = filedialog.askopenfilename(
            title="Select Reference Text",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not txt_path:
            return
        
        self.ref_wav = wav_path
        self.ref_txt = txt_path
        self.status_label.config(text="Custom voice loaded")
        
        # Reload reference in background
        threading.Thread(target=self._reload_reference, daemon=True).start()
    
    def save_current_voice(self):
        if not self.ref_wav or not self.ref_txt:
            messagebox.showwarning("No Voice", "Please load a voice first")
            return
        
        name = tk.simpledialog.askstring("Save Voice", "Enter a name for this voice:")
        if not name:
            return
        
        if self.profile_manager.add_profile(name, self.ref_wav, self.ref_txt):
            self.load_voice_profiles()
            self.voice_combo.set(name)
            self.status_label.config(text=f"Saved voice: {name}")
        else:
            messagebox.showerror("Error", "Failed to save voice profile")
    
    def delete_voice(self):
        selected = self.voice_combo.get()
        if selected == "Default":
            messagebox.showinfo("Info", "Cannot delete default voice")
            return
        
        if messagebox.askyesno("Delete Voice", f"Delete voice profile '{selected}'?"):
            if self.profile_manager.delete_profile(selected):
                self.load_voice_profiles()
                self.voice_combo.current(0)
                self.status_label.config(text=f"Deleted voice: {selected}")
    
    def _split_into_sentence_chunks(self, text: str, max_chars: int = 150) -> List[str]:
        """Split text into chunks of up to max_chars, ending at sentence boundaries"""
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # Calculate the end position for this chunk
            chunk_end = min(current_pos + max_chars, text_length)
            
            # If we're at the end of the text, take everything
            if chunk_end == text_length:
                chunk = text[current_pos:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Look for the last sentence ending before chunk_end
            chunk_text = text[current_pos:chunk_end]
            
            # Find all sentence endings (. ! ?) in this chunk
            sentence_endings = []
            for match in re.finditer(r'[.!?]\s', chunk_text):
                sentence_endings.append(match.end())
            
            if sentence_endings:
                # Use the last sentence ending
                actual_end = current_pos + sentence_endings[-1]
                chunk = text[current_pos:actual_end].strip()
                if chunk:
                    chunks.append(chunk)
                current_pos = actual_end
            else:
                # No sentence ending found, look for other break points (comma, semicolon)
                break_points = []
                for match in re.finditer(r'[,;:]\s', chunk_text):
                    break_points.append(match.end())
                
                if break_points:
                    actual_end = current_pos + break_points[-1]
                    chunk = text[current_pos:actual_end].strip()
                    if chunk:
                        chunks.append(chunk)
                    current_pos = actual_end
                else:
                    # No good break point, just take up to max_chars at a word boundary
                    words = chunk_text.split()
                    if len(words) > 1:
                        # Take all but the last word to avoid cutting mid-word
                        safe_text = ' '.join(words[:-1])
                        actual_end = current_pos + len(safe_text)
                        chunk = text[current_pos:actual_end].strip()
                        if chunk:
                            chunks.append(chunk)
                        current_pos = actual_end
                    else:
                        # Single long word, just take it
                        chunk = chunk_text.strip()
                        if chunk:
                            chunks.append(chunk)
                        current_pos = chunk_end
        
        return chunks
    
    def _calculate_eta(self, completed: int, total: int) -> str:
        """Calculate ETA with 95% confidence interval"""
        if completed == 0 or len(self.chunk_times) < 2:
            return "--:--"
        
        remaining = total - completed
        
        # Calculate mean and standard deviation
        mean_time = statistics.mean(self.chunk_times)
        
        if len(self.chunk_times) >= 2:
            stdev = statistics.stdev(self.chunk_times)
        else:
            stdev = 0
        
        # 95% CI: mean ± 1.96 * (stdev / sqrt(n))
        n = len(self.chunk_times)
        margin = 1.96 * (stdev / (n ** 0.5)) if n > 1 else 0
        
        # Calculate ETA
        eta_mean = remaining * mean_time
        eta_lower = remaining * max(0, mean_time - margin)
        eta_upper = remaining * (mean_time + margin)
        
        # Format times
        def format_time(seconds):
            if seconds < 60:
                return f"{int(seconds)}s"
            elif seconds < 3600:
                mins = int(seconds / 60)
                secs = int(seconds % 60)
                return f"{mins}m {secs}s"
            else:
                hours = int(seconds / 3600)
                mins = int((seconds % 3600) / 60)
                return f"{hours}h {mins}m"
        
        # Return ETA with confidence interval
        if margin > 0 and stdev > 0:
            return f"{format_time(eta_mean)} (±{format_time(margin * remaining)})"
        else:
            return format_time(eta_mean)
    
    def generate_audio(self):
        if self.is_generating:
            return
            
        text = self.textbox.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text first")
            return
        
        # Check if resuming or starting fresh
        if self.resuming_session:
            chunks = self.current_chunks
            start_index = len(self.completed_chunk_indices)
            print(f"[tts] Resuming from chunk {start_index}/{len(chunks)}")
        else:
            # Split text into 150-character chunks ending at sentences
            # (reduced from 300 to avoid token limit with reference text)
            chunks = self._split_into_sentence_chunks(text, max_chars=150)
            self.current_chunks = chunks
            self.completed_chunk_indices = []
            self.current_audio_sequence = []
            start_index = 0
            
            if not chunks:
                messagebox.showwarning("No Text", "Could not parse any text.")
                return
            
            print(f"[tts] Split into {len(chunks)} chunks (max 150 chars each)")
        
        self.is_generating = True
        self.chunk_times = []
        self.start_time = time.time()
        self.generate_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        
        if not self.resuming_session:
            self.download_btn.config(state="disabled")
            self.progress_var.set(0)
            self.progress_label.config(text=f"0/{len(chunks)}")
        
        self.eta_label.config(text="Calculating...")
        self.resuming_session = False
        
        # Start generation thread
        self.generation_thread = threading.Thread(
            target=self._generate_audio_thread, 
            args=(chunks, start_index), 
            daemon=True
        )
        self.generation_thread.start()
    
    def _split_paragraph_intelligently(self, paragraph: str) -> List[str]:
        """Split paragraph into smaller chunks if needed"""
        # First try by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [paragraph]

        # If only one sentence, split by half
        if len(sentences) == 1:
            mid = len(paragraph) // 2
            # Find a good break point (space, comma, etc.)
            for offset in range(50):  # Look within 50 chars of midpoint
                if mid + offset < len(paragraph) and paragraph[mid + offset] in ' ,-;:':
                    return [paragraph[:mid + offset].strip(), paragraph[mid + offset:].strip()]
                if mid - offset > 0 and paragraph[mid - offset] in ' ,-;:':
                    return [paragraph[:mid - offset].strip(), paragraph[mid - offset:].strip()]
            # No good break point, just split at midpoint
            return [paragraph[:mid].strip(), paragraph[mid:].strip()]

        return sentences

    def _generate_single_chunk(self, chunk: str, chunk_index: int):
        """Generate a single chunk (called by parallel workers) - THREAD-SAFE via lock"""
        chunk_start = time.time()
        try:
            # Phonemizer access is protected by lock in synthesize()
            audio_data = self.tts_engine.synthesize(chunk, skip_gc=True)
            return (chunk_start, audio_data, True)
        except TokenLimitError:
            # Try splitting if too long
            print(f"[tts] Chunk {chunk_index} too long, attempting to split...")
            sub_chunks = self._split_paragraph_intelligently(chunk)
            if len(sub_chunks) <= 1:
                print(f"[tts] Could not split chunk {chunk_index}, skipping")
                return (chunk_start, None, False)

            # Process sub-chunks
            sub_audio = []
            for sub_chunk in sub_chunks:
                try:
                    audio = self.tts_engine.synthesize(sub_chunk, skip_gc=True)
                    if audio:
                        sub_audio.append(audio)
                except Exception as e:
                    print(f"[tts] Sub-chunk failed: {e}")

            if sub_audio:
                # Combine sub-chunks into single audio
                import soundfile as sf
                import numpy as np
                combined_data = []
                for audio_bytes in sub_audio:
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    combined_data.append(data)
                combined = np.concatenate(combined_data)
                tmp = io.BytesIO()
                sf.write(tmp, combined, 24000, format="WAV")
                tmp.seek(0)
                return (chunk_start, tmp.getvalue(), True)
            else:
                return (chunk_start, None, False)

        except Exception as e:
            print(f"[tts] Error in chunk {chunk_index}: {e}")
            traceback.print_exc()
            return (chunk_start, None, False)
    
    def _generate_audio_thread(self, chunks, start_index=0):
        try:
            # Ensure model and reference are loaded
            self.tts_engine.initialize_model()
            self.tts_engine.load_reference(self.ref_wav, self.ref_txt)

            total_chunks = len(chunks)
            processed_count = len(self.completed_chunk_indices)

            # Create a lock for thread-safe operations on shared data
            import threading as th
            lock = th.Lock()

            # Use parallel processing with ThreadPoolExecutor
            # Phonemizer is protected by lock in synthesize(), so this is now safe
            batch_size = self.max_workers * 3  # Process 3 batches at a time per worker

            for batch_start in range(start_index, total_chunks, batch_size):
                if not self.is_generating:
                    # Save session on cancel
                    self.session_manager.save_session(
                        self.textbox.get("1.0", tk.END).strip(),
                        self.current_chunks,
                        self.completed_chunk_indices,
                        self.ref_wav,
                        self.ref_txt,
                        self.voice_combo.get()
                    )
                    self.root.after(0, lambda: self.status_label.config(text="Generation cancelled - session saved"))
                    break

                batch_end = min(batch_start + batch_size, total_chunks)
                batch_indices = list(range(batch_start, batch_end))

                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all chunks in batch
                    future_to_index = {}
                    for i in batch_indices:
                        if i in self.completed_chunk_indices:
                            continue  # Skip already completed chunks
                        chunk = chunks[i]
                        future = executor.submit(self._generate_single_chunk, chunk, i)
                        future_to_index[future] = i

                    # Process completed chunks as they finish
                    for future in as_completed(future_to_index):
                        if not self.is_generating:
                            break

                        i = future_to_index[future]
                        try:
                            chunk_start_time, audio_data, success = future.result()

                            if success and audio_data:
                                chunk_end = time.time()
                                chunk_time = chunk_end - chunk_start_time

                                with lock:
                                    self.chunk_times.append(chunk_time)
                                    self.current_audio_sequence.append((i, audio_data))
                                    self.session_manager.save_audio_chunk(i, audio_data)
                                    self.completed_chunk_indices.append(i)
                                    processed_count += 1

                                progress = (processed_count / total_chunks) * 100
                                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                                self.root.after(0, lambda count=processed_count, total=total_chunks:
                                              self.progress_label.config(text=f"{count}/{total}"))

                                # Update ETA
                                eta_str = self._calculate_eta(processed_count, total_chunks)
                                self.root.after(0, lambda eta=eta_str: self.eta_label.config(text=eta))

                                # Update status
                                self.root.after(0, lambda count=processed_count, total=total_chunks:
                                              self.status_label.config(
                                                  text=f"Processing in parallel ({count}/{total})..."))

                                # Enable download button after first successful chunk
                                if processed_count == 1:
                                    self.root.after(0, lambda: self.download_btn.config(state="normal"))

                        except Exception as e:
                            print(f"[tts] Error processing chunk {i}: {e}")
                            traceback.print_exc()

                # Garbage collection after each batch
                gc.collect()
                print(f"[memory] Batch {batch_start}-{batch_end} complete, GC performed")

            if self.is_generating and self.current_audio_sequence:
                # Sort audio sequence by index to maintain correct order
                with lock:
                    self.current_audio_sequence.sort(key=lambda x: x[0])
                    self.current_audio_sequence = [audio for _, audio in self.current_audio_sequence]

                total_time = time.time() - self.start_time
                chunks_per_sec = len(self.current_audio_sequence) / total_time if total_time > 0 else 0
                self.root.after(0, lambda t=total_time, cps=chunks_per_sec: self.status_label.config(
                    text=f"Complete! {len(self.current_audio_sequence)} chunks in {int(t)}s ({cps:.2f} chunks/sec)"))
                self.root.after(0, lambda: self.eta_label.config(text="Complete!"))
                # Clear session on successful completion
                self.session_manager.clear_session()
            elif not self.current_audio_sequence:
                self.root.after(0, lambda: self.status_label.config(text="No audio generated."))
                self.root.after(0, lambda: self.eta_label.config(text="--:--"))

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Generation failed: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.eta_label.config(text="Error"))
            # Save session on error
            self.session_manager.save_session(
                self.textbox.get("1.0", tk.END).strip(),
                self.current_chunks,
                self.completed_chunk_indices,
                self.ref_wav,
                self.ref_txt,
                self.voice_combo.get()
            )
        finally:
            self.is_generating = False
            self.root.after(0, lambda: self.generate_btn.config(state="normal"))
            self.root.after(0, lambda: self.cancel_btn.config(state="disabled"))
    
    def cancel_generation(self):
        self.is_generating = False
        self.cancel_btn.config(state="disabled")
        self.status_label.config(text="Cancelling generation...")
        self.eta_label.config(text="Cancelled")
    
    def download_audio(self):
        if not self.current_audio_sequence:
            messagebox.showwarning("No Audio", "No audio has been generated yet.")
            return
        
        try:
            import soundfile as sf
            import numpy as np
            
            # Combine all chunks into a single audio file
            all_audio_data = []
            for chunk_bytes in self.current_audio_sequence:
                data, sr = sf.read(io.BytesIO(chunk_bytes))
                all_audio_data.append(data)
            
            if not all_audio_data:
                messagebox.showwarning("No Audio", "No audio data to save.")
                return
            
            combined_audio = np.concatenate(all_audio_data)
            
            # Ask user where to save
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialfile=f"neutts_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            if filename:
                sf.write(filename, combined_audio, 24000)
                self.status_label.config(text=f"Saved to: {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Audio saved successfully!\n{os.path.basename(filename)}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save audio: {e}")

# ==============================
# Entry
# ==============================
def main():
    print("=" * 60)
    print("NeuTTS-Air GUI - OPTIMIZED VERSION")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print("\nOptimizations enabled:")
    print("  ✓ GPU auto-detection (CUDA/MPS) for codec acceleration")
    print("  ✓ Parallel chunk processing (2-4 workers)")
    print("  ✓ Thread-safe phonemizer (protected by lock)")
    print("  ✓ Reduced garbage collection overhead")
    print("  ✓ Batch processing for better memory management")
    print("=" * 60)
    
    # Try to import tkinter.simpledialog for the save dialog
    try:
        import tkinter.simpledialog
    except ImportError:
        import tkinter.simpledialog as simpledialog
        tk.simpledialog = simpledialog
    
    set_phonemizer_env()
    
    try:
        ensure_all_dependencies()
    except Exception as e:
        print(f"[error] Failed to install dependencies: {e}")
        print("[info] Continuing anyway - some features may not work")
    
    set_phonemizer_env()
    root = tk.Tk()
    app = NeuTTSGui(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
