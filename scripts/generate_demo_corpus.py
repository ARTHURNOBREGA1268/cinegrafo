from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "examples" / "corpus"

MOVIE_PREFIXES = [
    "A Cidade", "O Jardim", "A Última", "As Sombras", "O Segredo",
    "Maré", "Estação", "Cartas", "O Arquivo", "A Ponte",
]
MOVIE_SUFFIXES = [
    "de Vidro", "das Cinzas", "do Norte", "de Saturno", "dos Relógios",
]
SERIES_PREFIXES = [
    "Os Navegantes", "Diário de", "As Crônicas", "O Observatório",
    "Ventos de", "A Academia", "Os Guardiões", "Memórias de",
    "Horizonte", "O Círculo",
]
SERIES_SUFFIXES = [
    "Aurora", "Orion", "Lunária", "Valedouro", "Ametista",
]
MOVIES = [f"{a} {b}" for a in MOVIE_PREFIXES for b in MOVIE_SUFFIXES]
SERIES = [f"{a} {b}" for a in SERIES_PREFIXES for b in SERIES_SUFFIXES]
GENRES = [
    "aventura", "ficção científica", "mistério", "fantasia", "drama",
    "comédia", "suspense", "animação", "romance", "documentário",
]
FRANCHISES = [
    "Mapas do Infinito", "Constelação de Jade", "Lendas do Horizonte",
    "Arquivos de Ametista", "Travessia de Órion", "Cidades da Aurora",
    "Relatos do Mar Antigo", "Círculo das Estações", "Portais de Safira",
    "Crônicas do Vento Norte",
]
FIRST_NAMES = [
    "Alina", "Bento", "Caio", "Dara", "Elisa", "Fábio", "Gael", "Helena",
    "Ícaro", "Joana", "Kai", "Lia", "Mauro", "Nina", "Otávio", "Paula",
    "Ravi", "Sofia", "Tomás", "Yara",
]
DIRECTOR_SURNAMES = ["Valença", "Nogueira", "Saldanha", "Medeiros", "Lacerda"]
ACTOR_SURNAMES = ["Vasconcelos", "Tavares", "Bittencourt", "Figueira", "Montalvão"]
CHARACTER_FIRST = ["Aren", "Belis", "Ciro", "Dália", "Eron", "Fira", "Garen", "Halia", "Ilan", "Jori"]
CHARACTER_LAST = ["Vento", "Sol", "Mar", "Lume", "Vale", "Névoa", "Cedro", "Prisma", "Duna", "Estrela"]

def person(first_index: int, surnames: list[str]) -> str:
    return f"{FIRST_NAMES[first_index % len(FIRST_NAMES)]} {surnames[first_index // len(FIRST_NAMES)]}"

def make_record(index: int, work_type: str, record_number: int) -> str:
    title = (MOVIES if work_type == "Movie" else SERIES)[index]
    entity_index = index + (50 if work_type == "Series" else 0)
    director = person(entity_index, DIRECTOR_SURNAMES)
    actor = person(entity_index, ACTOR_SURNAMES)
    character = f"{CHARACTER_FIRST[entity_index % 10]} {CHARACTER_LAST[entity_index // 10]}"
    genre = GENRES[(entity_index * 3) % len(GENRES)]
    franchise = FRANCHISES[(entity_index * 7) % len(FRANCHISES)]
    year = 1988 + ((entity_index * 7) % 39)
    if work_type == "Movie":
        facts = (
            f"O filme fictício “{title}” estreou no ano inventado de {year}, tem gênero {genre} "
            f"e integra a franquia fictícia “{franchise}”. "
            f"{director} dirigiu o filme “{title}”. "
            f"{actor} atuou no filme “{title}” como a personagem fictícia “{character}”."
        )
    else:
        facts = (
            f"A série fictícia “{title}” estreou no ano inventado de {year}, tem gênero {genre} "
            f"e integra a franquia fictícia “{franchise}”. "
            f"{director} criou a série “{title}”. "
            f"{actor} atuou na série “{title}” como a personagem fictícia “{character}”."
        )
    return f"Registro {record_number:03d} — {facts}"

def write_file(filename: str, entries: list[str]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / filename).write_text("\n\n".join(entries) + "\n", encoding="utf-8")

if __name__ == "__main__":
    write_file("01_catalogo.txt", [make_record(i, "Movie", i + 1) for i in range(25)])
    write_file("02_series.txt", [make_record(i, "Series", i + 1) for i in range(25, 50)])
    write_file("03_producao.txt", [make_record(i, "Movie", i + 26) for i in range(25, 50)])
    write_file("04_guia.txt", [make_record(i, "Series", i + 76) for i in range(25)])
    print(f"Gerados {len(MOVIES) + len(SERIES)} títulos sintéticos únicos em 4 arquivos.")
