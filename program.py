import random
import numpy as np
import openpyxl

# Ścieżka do pliku Excel
sciezka_pliku = r"C:\Users\ostat\Desktop\SIDane.xlsx"


# Wczytywanie danych z pliku Excel
def wczytaj_dane(sciezka_pliku):
    skoroszyt = openpyxl.load_workbook(sciezka_pliku)
    arkusz = skoroszyt.active
    dane = []
    for wiersz in arkusz.iter_rows(min_row=2, values_only=True):
        try:
            dane.append(list(wiersz[:8]))  # Pobieranie pierwszych 8 kolumn
        except ValueError:
            pass
    skoroszyt.close()
    return dane


# Podział danych na zbiór treningowy i testowy
def podziel_dane(dane, wspolczynnik_treningowy):
    rozmiar_treningowy = int(len(dane) * wspolczynnik_treningowy)
    dane_treningowe = random.sample(dane, rozmiar_treningowy)
    dane_testowe = [d for d in dane if d not in dane_treningowe]
    return dane_treningowe, dane_testowe


# Inicjalizacja sieci neuronowej
def inicjalizuj_siec():
    np.random.seed(0)
    siec = np.random.uniform(-1, 1, (5, 1))
    return siec


# Funkcja sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Obliczanie wyniku dla danej instancji
def oblicz_wynik(siec, instancja):
    suma_wazona = np.dot(instancja[:5], siec)
    wynik = sigmoid(suma_wazona)
    return wynik[0]


# Ocena przystosowania sieci na podstawie danych treningowych
def ocen_przystosowanie(siec, podane_dane):
    poprawne_przewidywania = 0
    for dane in podane_dane:
        geny = dane[:5]  # Pierwsze 5 kolumn to geny
        poprawna_odpowiedz = dane[5]  # 6. kolumna to poprawna odpowiedź
        przewidziany_wynik = oblicz_wynik(siec, geny)
        przewidziany_wynik_bin = 1 if przewidziany_wynik > 0.5 else 0
        if przewidziany_wynik_bin == poprawna_odpowiedz:
            poprawne_przewidywania += 1
    dokladnosc = (poprawne_przewidywania / len(podane_dane)) * 100

    return dokladnosc


# Implementacja algorytmu genetycznego
def algorytm_genetyczny(rozmiar_populacji, liczba_generacji, prawdop_mutacji, dane_treningowe):
    populacja = [(np.random.uniform(-1, 1, (5, 1)), 0) for _ in range(rozmiar_populacji)]
    populacja = [(siec, ocen_przystosowanie(siec, dane_treningowe)) for siec, _ in populacja]

    print("Populacja:")
    for osobnik in populacja:
        siec, przystosowanie = osobnik
        print(f"Sieć:\n{siec}")
        print(f"Przystosowanie:\n{przystosowanie}")

    najlepsze_wyniki_przystosowania = []
    for generacja in range(liczba_generacji):
        potomstwo = []

        for i in range(rozmiar_populacji):
            print(f"\nGeneracja: {generacja + 1} | Potomstwo: {i + 1}")

            # Selekcja turniejowa
            przystosowanie_populacji = [przystosowanie for _, przystosowanie in populacja]
            populacja.sort(key=lambda x: x[1], reverse=True)  # Sortowanie populacji według wyników przystosowania

            rodzic1, przystosowanie_rodzic1 = populacja[0]
            rodzic2, przystosowanie_rodzic2 = populacja[1]

            print("Przystosowanie rodzica 1:", przystosowanie_rodzic1)
            print("Przystosowanie rodzica 2:", przystosowanie_rodzic2)

            # Krzyżowanie jednopunktowe
            punkt_podzialu = random.randint(1, len(rodzic1) - 1)
            potomstwo_sieci = np.concatenate((rodzic1[:punkt_podzialu], rodzic2[punkt_podzialu:]))

            print(f"\nEtap oceny przystosowania dla potomstwa {i + 1}")
            print(f"Rodzic 1:")
            print(rodzic1)
            print(f"Rodzic 2:")
            print(rodzic2)
            print(f"Potomstwo:")
            print(potomstwo_sieci)

            # Mutacja
            for j in range(len(potomstwo_sieci)):
                if random.random() < prawdop_mutacji:
                    potomstwo_sieci[j] += random.uniform(-0.1, 0.1)
                    print("Mutacja! Nowa wartość:", potomstwo_sieci[j])

            # Obliczenie przystosowania dla potomstwa
            przystosowanie = ocen_przystosowanie(potomstwo_sieci, dane_treningowe)
            potomstwo.append((potomstwo_sieci, przystosowanie))
            print(f"Przystosowanie: {przystosowanie:.2f}%")

        # Aktualizacja populacji
        populacja += potomstwo
        populacja.sort(key=lambda x: x[1], reverse=True)

        # Zachowanie połowy najlepszych sieci do następnej generacji
        populacja_najlepsze = populacja[:rozmiar_populacji // 2]

        # Wygenerowanie nowej populacji losowo
        populacja_nowa = [(np.random.uniform(-1, 1, (5, 1)), 0) for _ in range(rozmiar_populacji - (rozmiar_populacji // 2))]
        populacja_nowa = [(siec, ocen_przystosowanie(siec, dane_treningowe)) for siec, _ in populacja_nowa]

        # Połączenie populacji najlepszych i nowych sieci
        populacja = populacja_najlepsze + populacja_nowa

        # Zapisanie najlepszego wyniku dla bieżącej generacji
        najlepsze_przystosowanie = populacja[0][1]
        najlepsze_wyniki_przystosowania.append(najlepsze_przystosowanie)
        print(f"\nGeneracja {generacja + 1}: Najlepsze przystosowanie = {najlepsze_przystosowanie:.2f}%")

    najlepsza_siec = populacja[0][0]
    return najlepsza_siec, najlepsze_wyniki_przystosowania


# Użycie algorytmu genetycznego do nauki sieci neuronowej
rozmiar_populacji = 10
liczba_generacji = 25
prawdop_mutacji = 0.3
wspolczynnik_treningowy = 0.4

dane = wczytaj_dane(sciezka_pliku)
dane_treningowe, dane_testowe = podziel_dane(dane, wspolczynnik_treningowy)

najlepsza_siec, najlepsze_wyniki_przystosowania = algorytm_genetyczny(rozmiar_populacji, liczba_generacji, prawdop_mutacji, dane_treningowe)

# Wyświetlanie meczów, przewidywanych wyników i poprawnych wyników na danych testowych
print("\nMecze, przewidywane wyniki i poprawne wyniki na danych testowych:")
for dane in dane_testowe:
    pkt_w_sezonie, wartosc, u_siebie, h2h, tabela, wygrana, sezon, przeciwnik = dane
    przewidziany_wynik = oblicz_wynik(najlepsza_siec, dane)
    przewidziany_wynik_bin = "Wygrana" if przewidziany_wynik > 0.5 else "Przegrana"
    poprawny_wynik = "Wygrana" if wygrana == 1 else "Przegrana"
    print(
        f"Mecz: {przeciwnik} w sezonie {sezon} - Przewidywany wynik: {przewidziany_wynik_bin} (Poprawny wynik: {poprawny_wynik})")

# Obliczanie procentowej skuteczności na danych testowych
dokladnosc = ocen_przystosowanie(najlepsza_siec, dane_testowe)
print(f"\nSkuteczność sieci neuronowej na danych testowych: {dokladnosc:.2f}%")

# Wyświetlanie najlepszego wyniku dla każdej generacji
print("\nNajlepszy wynik dla każdej generacji:")
for generacja, najlepsze_przystosowanie in enumerate(najlepsze_wyniki_przystosowania):
    print(f"Generacja {generacja + 1}: Najlepsze przystosowanie = {najlepsze_przystosowanie:.2f}%")

