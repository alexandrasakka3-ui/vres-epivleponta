from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from data import search as search_data


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
INSTITUTIONS = ["Θράκη", "Κρήτη", "Μακεδονία"]
INSTITUTION_COLORS = {
    "Θράκη": "#e76f51",
    "Κρήτη": "#2a9d8f",
    "Μακεδονία": "#457b9d",
    "Άλλο": "#8d99ae",
}
EXAMPLE_QUERIES = [
    "Ανανεώσιμες πηγές ενέργειας",
    "Μηχανική μάθηση στην ιατρική",
    "Εκπαίδευση παιδιών με ειδικές ανάγκες",
    "Τουρισμός και βιώσιμη ανάπτυξη",
]

st.set_page_config(page_title="Βρες Επιβλέποντα", page_icon="🎓", layout="wide")


@st.cache_resource
def load_search_resources():
    return search_data._load_resources()


@st.cache_resource
def load_graph_resources():
    edges, people = search_data._load_graph()
    people_counts = pd.read_parquet(DATA_DIR / "people.parquet")
    return edges, people, people_counts


def set_example_query(query: str) -> None:
    st.session_state["interest_query"] = query


def text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def committee_members(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        members = value
    elif hasattr(value, "tolist"):
        members = value.tolist()
    else:
        members = [value]
    return [
        name
        for item in members
        if (name := text_value(item))
        and name.casefold() not in {"δεν διατίθεται", "δε διατίθεται"}
    ]


def abstract_text(value: object) -> str:
    abstract = text_value(value)
    return re.sub(
        r"^Περίληψη(?:\s*\(Abstract\))?(?=$|[\s:：])\s*[:：]?\s*",
        "",
        abstract,
        count=1,
        flags=re.IGNORECASE,
    )


def institution_for_node(institutions: object) -> str:
    if isinstance(institutions, str):
        values = [institutions]
    elif institutions is None:
        values = []
    elif hasattr(institutions, "tolist"):
        values = institutions.tolist()
    else:
        values = list(institutions)
    return next(
        (value for value in INSTITUTIONS if value in values),
        "Άλλο",
    )


def render_network(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        st.info("Δεν υπάρχουν προτεινόμενοι επιβλέποντες για να εμφανιστεί δίκτυο.")
        return

    seed_ids = recommendations["supervisor_id"].astype(str).tolist()
    nodes, edges = search_data.neighborhood(seed_ids, max_nodes=40)
    if edges.empty:
        st.info("Δεν βρέθηκαν κοινές επιτροπές για αυτούς τους επιβλέποντες.")
        return

    _, _, people_counts = load_graph_resources()
    counts = people_counts.set_index("person_id")["n_as_supervisor"].to_dict()
    recommended_ids = set(seed_ids)
    network = Network(
        height="620px",
        width="100%",
        directed=False,
        bgcolor="#ffffff",
        font_color="#263238",
        cdn_resources="in_line",
    )
    network.barnes_hut()

    for node in nodes.itertuples(index=False):
        person_id = str(node.person_id)
        name = text_value(node.canonical_name) or person_id
        work_count = int(counts.get(person_id, 0))
        is_recommended = person_id in recommended_ids
        network.add_node(
            person_id,
            label=name,
            title=f"{name}<br>Εργασίες ως επιβλέπων: {work_count}",
            color=INSTITUTION_COLORS[institution_for_node(node.institutions)],
            size=15 + min(work_count, 100) ** 0.5 * 2,
            font={
                "size": 18 if is_recommended else 14,
                "bold": is_recommended,
            },
        )

    for edge in edges.itertuples(index=False):
        weight = int(edge.weight)
        network.add_edge(
            str(edge.person_a),
            str(edge.person_b),
            value=weight,
            title=f"Κοινές επιτροπές: {weight}",
        )

    components.html(network.generate_html(), height=640, scrolling=True)
    legend = " · ".join(
        f"<span style='color:{INSTITUTION_COLORS[name]}'>●</span> {name}"
        for name in INSTITUTIONS
    )
    st.markdown(legend, unsafe_allow_html=True)


def render_results(
    results: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    works_tab, supervisors_tab, network_tab = st.tabs(
        ["Σχετικές εργασίες", "Προτεινόμενοι επιβλέποντες", "Δίκτυο επιβλεπόντων"]
    )

    with works_tab:
        if results.empty:
            st.info("Δεν βρέθηκαν σχετικές εργασίες για τα επιλεγμένα ιδρύματα.")
        else:
            for index, row in enumerate(results.head(10).itertuples(index=False), 1):
                st.markdown(f"**{index}.**")
                title = text_value(row.title) or "Χωρίς τίτλο"
                url = text_value(row.url)
                if url:
                    st.link_button(title, url)
                else:
                    st.markdown(f"**{title}**")

                details = [
                    f"**Ίδρυμα:** {text_value(row.institution) or '—'}",
                    f"**Έτος:** {int(row.year) if pd.notna(row.year) else '—'}",
                    f"**Επιβλέπων:** {text_value(row.supervisor_name) or '—'}",
                    f"**Score:** {float(row.score):.4f}",
                ]
                members = committee_members(row.committee_names)
                if members:
                    details.append(f"**Μέλη επιτροπής:** {' · '.join(members)}")
                st.markdown("  \n".join(details))

                abstract = abstract_text(row.abstract)
                if abstract:
                    preview = abstract[:300]
                    if len(abstract) > 300:
                        preview += "…"
                    st.write(preview)
                    with st.expander("Ολόκληρη περίληψη"):
                        st.write(abstract)
                st.divider()

    with supervisors_tab:
        if recommendations.empty:
            st.info("Δεν βρέθηκαν επιβλέποντες με σχετικές εργασίες.")
        else:
            table = recommendations.head(10).copy()
            table["Ιδρύματα"] = table["institutions"].map(", ".join)
            table["Πλήθος σχετικών εργασιών"] = table["related_work_count"]
            table["Τρεις σχετικοί τίτλοι"] = table["top_titles"].map(
                lambda titles: " · ".join(str(title) for title in titles)
            )
            st.dataframe(
                table[
                    [
                        "supervisor_name",
                        "Ιδρύματα",
                        "Πλήθος σχετικών εργασιών",
                        "Τρεις σχετικοί τίτλοι",
                    ]
                ].rename(columns={"supervisor_name": "Όνομα"}),
                hide_index=True,
                width="stretch",
            )

    with network_tab:
        render_network(recommendations.head(10))


with st.sidebar:
    st.header("Παραδείγματα ερωτημάτων")
    for index, example in enumerate(EXAMPLE_QUERIES):
        st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
            on_click=set_example_query,
            args=(example,),
        )

    selected_institutions = st.multiselect(
        "Ιδρύματα",
        options=INSTITUTIONS,
        default=INSTITUTIONS,
    )

    st.header("Σχετικά")
    st.markdown(
        """
        **Πηγή δεδομένων:** glossAPI/libduth, elocus και psepheda.

        Τα δεδομένα διατίθενται με άδεια **CC BY-NC-ND 4.0** και προορίζονται
        για μη εμπορική χρήση.

        **Περιορισμοί**
        - Η ενοποίηση ονομάτων είναι αυτόματη και μπορεί να έχει λάθη.
        - Οι επιτροπές είναι διαθέσιμες μόνο για μέρος των εργασιών, κυρίως
          από τη Θράκη και την Κρήτη.
        - Τα αποτελέσματα βασίζονται στη σημασιολογική ομοιότητα και μπορεί
          να περιλαμβάνουν ψευδώς σχετικές εργασίες.
        """
    )

st.title("Βρες Επιβλέποντα")
st.write(
    "Περιέγραψε το ερευνητικό σου ενδιαφέρον για να ανακαλύψεις σχετικές "
    "εργασίες και πιθανούς επιβλέποντες."
)

with st.form("search_form"):
    query = st.text_area(
        "Ερευνητικό ενδιαφέρον",
        placeholder="π.χ. βιώσιμη ανάπτυξη και τουρισμός",
        height=140,
        key="interest_query",
    )
    submitted = st.form_submit_button("Αναζήτηση", type="primary")

if submitted:
    if not query.strip():
        st.warning("Γράψε πρώτα το ερευνητικό σου ενδιαφέρον.")
    elif not selected_institutions:
        st.warning("Επίλεξε τουλάχιστον ένα ίδρυμα.")
    else:
        with st.spinner("Αναζητούνται σχετικές εργασίες και επιβλέποντες..."):
            load_search_resources()
            results = search_data.search(
                query.strip(),
                institutions=selected_institutions,
                top_k=100,
            )
            recommendations = search_data.recommend_supervisors(
                results,
                top_n=10,
            )
            st.session_state["search_results"] = results
            st.session_state["supervisor_recommendations"] = recommendations
            st.session_state["searched_query"] = query.strip()

if "search_results" in st.session_state:
    st.caption(f"Ερώτημα: {st.session_state['searched_query']}")
    render_results(
        st.session_state["search_results"],
        st.session_state["supervisor_recommendations"],
    )
