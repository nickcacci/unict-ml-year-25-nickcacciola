import torch

GTSRB_CLASSES = {
0: "Limite di velocità (20 km/h)",
1: "Limite di velocità (30 km/h)",
2: "Limite di velocità (50 km/h)",
3: "Limite di velocità (60 km/h)",
4: "Limite di velocità (70 km/h)",
5: "Limite di velocità (80 km/h)",
6: "Fine limite di velocità (80 km/h)",
7: "Limite di velocità (100 km/h)",
8: "Limite di velocità (120 km/h)",
9: "Divieto di sorpasso",
10: "Divieto di sorpasso per autocarri (>3.5t)",
11: "Diritto di precedenza al prossimo incrocio",
12: "Strada con diritto di precedenza",
13: "Dare precedenza",
14: "Fermarsi e dare precedenza (STOP)",
15: "Circolazione vietata in entrambe le direzioni",
16: "Transito vietato ai veicoli pesanti (>3.5t)",
17: "Senso vietato (Divieto di accesso)",
18: "Pericolo generico",
19: "Curva pericolosa a sinistra",
20: "Curva pericolosa a destra",
21: "Doppia curva pericolosa (la prima a sinistra)",
22: "Strada deformata o dissestata",
23: "Strada sdrucciolevole",
24: "Strettoia simmetrica o a destra",
25: "Lavori in corso",
26: "Semaforo / Segnale luminoso",
27: "Attraversamento pedonale",
28: "Attraversamento bambini / Scuola",
29: "Attraversamento ciclabile",
30: "Pericolo formazione ghiaccio / neve",
31: "Attraversamento animali selvatici",
32: "Fine di tutte le prescrizioni e limiti di velocità",
33: "Direzione obbligatoria a destra",
34: "Direzione obbligatoria a sinistra",
35: "Direzione obbligatoria dritto",
36: "Consentito dritto o a destra",
37: "Consentito dritto o a sinistra",
38: "Passaggio obbligatorio a destra",
39: "Passaggio obbligatorio a sinistra",
40: "Circolazione rotatoria obbligatoria",
41: "Fine divieto di sorpasso",
42: "Fine divieto di sorpasso per autocarri",
}   

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