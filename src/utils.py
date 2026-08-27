import torch

def verify_torch_gpu ():
    print("=== Verifica Setup PyTorch e GPU ===")
    print(f"Versione PyTorch: {torch.__version__}")
    print(f"CUDA Disponibile: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Versione CUDA integrata: {torch.version.cuda}")
        print(f"Numero di GPU rilevate: {torch.cuda.device_count()}")
        print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capacità Computazionale: {torch.cuda.get_device_capability(0)}")
        
        # Test allocazione di un tensore di prova in memoria GPU
        x = torch.ones((1000, 1000), device="cuda")
        print(f"Test allocazione tensore su GPU: RIUSCITO (Dispositivo: {x.device})")
    else:
        print("ATTENZIONE: PyTorch sta attualmente eseguendo su CPU.")