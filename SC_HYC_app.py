elif ticker_source == "Vordefiniert (eingebettet)":
    embedded = get_embedded_lists()
    combined = []
    if embedded:
        emb_choices = st.sidebar.multiselect(
            "Eingebettete Listen wählen",
            options=sorted(embedded.keys()),
            help="Im Code mitgelieferte Auszüge, jederzeit erweiterbar."
        )
        for nm in emb_choices:
            combined += (embedded.get(nm) or [])

    # Basisliste aus eingebetteten Sets
    base = _normalize_tickers(combined)

    # 👉 NEU: manuelle Ergänzungen nach Listen-Auswahl
    extra_embed = st.sidebar.text_input(
        "Weitere Ticker manuell hinzufügen (Komma-getrennt)",
        value="", key="extra_embed",
        help="Beispiel: AAPL, TSLA, BABA"
    )
    extras = _normalize_tickers([t for t in extra_embed.split(",") if t.strip()]) if extra_embed else []

    # Vereinen + Deduplizieren
    tickers_final = _normalize_tickers(base + extras)

    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker (vereint & dedupliziert): {len(tickers_final)}")

        shuffle_lists = st.sidebar.checkbox("Zufällig mischen", value=False, key="shuffle_embed")
        if shuffle_lists:
            import random
            random.seed(42); random.shuffle(tickers_final)

        max_n = st.sidebar.number_input(
            "Max. Anzahl (0 = alle)",
            min_value=0, max_value=len(tickers_final),
            value=min(50, len(tickers_final)), step=10, key="maxn_embed"
        )
        if max_n and max_n < len(tickers_final):
            tickers_final = tickers_final[:int(max_n)]

        # Multiselect zeigt auch die manuell hinzugefügten
        tickers_final = st.sidebar.multiselect(
            "Tickers aus den gewählten Listen", options=tickers_final, default=tickers_final
        )

        with st.sidebar.expander("Vorschau (erste 30)"):
            st.write(tickers_final[:30])

        st.sidebar.download_button(
            "Kombinierte Liste als CSV",
            pd.DataFrame({"ticker": tickers_final}).to_csv(index=False).encode("utf-8"),
            file_name="tickers_combined.csv", mime="text/csv"
        )
    else:
        st.sidebar.info("Noch keine Ticker ausgewählt/gefunden.")
