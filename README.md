# Ontologia grafu (Relacje i węzły)


**Węzły (Nodes)**
* `Kierunek` (właściwości: *stopien*, *wydzial*)
* `Dyscyplina_Naukowa`
* `Tryb_Studiow` (np. stacjonarne)
* `Olimpiada_Konkurs`
* `Egzamin_Wstepny`
* `Cykl_Rekrutacyjny`
* `Etap_Rekrutacji`
* `Statystyka_Rekrutacyjna`
* `Wzor_Rekrutacyjny`
* `Przedmiot_Maturalny`
* `Poziom_Egzaminu` (np. podstawa/rozszerzenie)

---

**Relacje i ich Właściwości**

**1. Struktura Kierunków i Progi**
* `(Kierunek) -[:NALEZY_DO]-> (Dyscyplina_Naukowa)`
* `(Kierunek) -[:OFEROWANY_JAKO]-> (Tryb_Studiow)`
* `(Tryb_Studiow) -[:MIAL_WYNIKI_W_ROKU]-> (Statystyka_Rekrutacyjna)` 
  * *Właściwości:* `rok_akademicki`, `prog_punktowy`, `limit_miejsc`

**2. Olimpiady i Zwolnienia z Rekrutacji**
* `(Olimpiada_Konkurs) -[:ZWALNIA_Z_REKRUTACJI_NA]-> (Kierunek)` 
  * *Właściwości:* `dziedzina`, `wymagany_tytul`, `typ_uprawnienia`

**3. Egzaminy Wstępne (II stopień)**
* `(Kierunek) -[:WYMAGA_EGZAMINU]-> (Egzamin_Wstepny)` 
  * *Właściwości:* `forma_i_sposob`, `uznanie_kierunkowego`, `data_I_cykl`, `data_II_cykl`

**4. Kalendarz i Terminy**
* `(Cykl_Rekrutacyjny) -[:SKLADA_SIE_Z]-> (Etap_Rekrutacji)` 
  * *Właściwości:* `data_rozpoczecia`, `data_zakonczenia`, `godzina_zakonczenia`, `wymaga_obecnosci`

**5. Wzory i Przeliczniki Maturalne**
* `(Kierunek) -[:WYKORZYSTUJE_WZOR]-> (Wzor_Rekrutacyjny)`
* `(Kierunek) -[:UWZGLEDNIA_W_M]-> (Przedmiot_Maturalny)` 
  * *Właściwości:* `waga_skladnika`
* `(Kierunek) -[:UWZGLEDNIA_W_P1]-> (Przedmiot_Maturalny)` 
  * *Właściwości:* `waga_skladnika`, `warunek_wykluczajacy`
* `(Kierunek) -[:UWZGLEDNIA_W_P2]-> (Przedmiot_Maturalny)` 
  * *Właściwości:* `waga_skladnika`
* `(Przedmiot_Maturalny) -[:ZDAWANY_NA]-> (Poziom_Egzaminu)` 
  * *Właściwości:* `przelicznik_poziomu`