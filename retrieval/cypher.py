CYPHER_QUERIES = {
    "matury": """
        MATCH (k:Kierunek {id: $kierunek})-[r:WYMAGA_PRZEDMIOTU]->(p:Przedmiot_Maturalny)
        RETURN k.id AS kierunek,
               collect({przedmiot: p.id, kolumna: r.kolumna}) AS przedmioty
    """,
    "progi": """
        MATCH (k:Kierunek)-[r:OFEROWANY_JAKO]->(t:Tryb_Studiow)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek, t.id AS tryb,
               r.prog_min AS prog_min, r.prog_max AS prog_max,
               r.miejsca  AS miejsca
        ORDER BY r.prog_max DESC
    """,
    "egzaminy": """
        MATCH (k:Kierunek)-[r:HONORUJE_EGZAMIN]->(e:Egzamin_Zawodowy)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek,
               collect({egzamin: e.id, kolumna: r.kolumna}) AS egzaminy
    """,
    "osiagniecia": """
        MATCH (k:Kierunek)-[r:DAJE_PUNKTY_ZA]->(o:Osiagniecie)
        WHERE $kierunek IS NULL OR k.id = $kierunek
        RETURN k.id AS kierunek,
               collect({osiagniecie: o.id, max_punktow: r.max_punktow}) AS osiagniecia
    """,
    "wzory": """
        MATCH (w:Wzor_Rekrutacyjny)
        RETURN w.id          AS id,
               w.zastosowanie AS zastosowanie,
               w.postac       AS postac,
               w.zmienne      AS zmienne
    """,
    "terminy": """
        MATCH (t:Termin_Rekrutacji)
        WHERE $kontekst IS NULL
           OR toLower(t.kontekst) CONTAINS toLower($kontekst)
           OR toLower(t.etap)     CONTAINS toLower($kontekst)
        RETURN t.tabela   AS tabela,
               t.kontekst AS kontekst,
               t.cykl     AS cykl,
               t.etap     AS etap,
               t.termin   AS termin,
               t.uwagi    AS uwagi
        ORDER BY t.cykl, t.etap
    """,
    "kierunki": """
        MATCH (k:Kierunek)
        RETURN collect(k.id) AS kierunki
    """,
    "_lista_przedmiotow": """
        MATCH (p:Przedmiot_Maturalny)
        RETURN collect(p.id) AS przedmioty
    """,
    "_lista_egzaminow": """
        MATCH (e:Egzamin_Zawodowy)
        RETURN collect(e.id) AS egzaminy
    """,
    "_lista_trybow": """
        MATCH (t:Tryb_Studiow)
        RETURN collect(t.id) AS tryby
    """,
}

WYMAGA_KIERUNKU = {"matury"}
