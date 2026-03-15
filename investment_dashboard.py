import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- Ticker ---
ASSETS = {
    "Welt":{'ticker': '^990100-USD-STRD', 'start': '1985-01-02'},
    "WDI":{'ticker': 'WDI.HM', 'start': '2017-10-01'},
    "Gold":{'ticker': 'GC=F', 'start': '2017-10-01'}
}

# --- Sidebar ---
st.sidebar.header("Investitionen")
var_First_Invest = st.sidebar.number_input("Start Investition (€)", value=1000)
var_Frequent_Invest = st.sidebar.number_input("Monatliche Investition (€)", value=50)

def format_de(n):
    return f"{n:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

@st.cache_data
def get_stock_data(ticker):

    data = yf.Ticker(ticker)

    # for ETF, if no trading day there is yahoo issue
    # other tickers do not have that long an history
    # date is therefore shifted manually
    for i in range(0,4):
        df = data.history(interval="1d", period="max", end=dt.datetime.now()-pd.Timedelta(days=i), auto_adjust=True)
        if not df.empty: break

    # remove timezone
    df.index = df.index.tz_localize(None)

    return pd.DataFrame(df['Close'])

def get_gold_data(df_gold):

    df_csv = pd.read_csv('gold_historical.csv', sep=';', parse_dates=['Date'])
    df_csv.set_index('Date', inplace=True)

    merge_date = df_gold.index.min()
    
    df_csv_pre = df_csv[df_csv.index < merge_date].copy()

    df_merged = pd.concat([df_csv_pre, df_gold])

    df_merged.sort_index(inplace=True)

    df_merged = df_merged.resample('D').ffill()

    return df_merged

def calc_logReturn(df):
    logR = np.log(df / df.shift(1)).dropna()
    return logR

def calc_historical_df(logR, var_First_Invest, var_Frequent_Invest):
    # regular investments are compounded in reverse
    # because it is a simple linear combination of single invests
    # output is a timeseries for plot

    # cumulated log returns
    cum_logR = np.insert(np.cumsum(logR.values.flatten()), 0, 0)[:-1]
    total_growth_factors = np.exp(cum_logR)
    
    # compound first invest
    path_start_invest = var_First_Invest * total_growth_factors
    
    # find month
    months = logR.index.month
    savings_impulses = np.zeros(len(logR))
    month_changes_mask = np.zeros(len(logR), dtype=bool)
    month_changes_mask[1:] = (months[1:] != months[:-1])
    
    # invest frequest investment in new month
    savings_impulses[month_changes_mask] = var_Frequent_Invest
    
    # discounting every investment to t0, then compound to final date
    discounted_savings = savings_impulses / total_growth_factors
    path_savings = np.cumsum(discounted_savings) * total_growth_factors
    
    return path_start_invest + path_savings

def calc_invest_df(logR, var_First_Invest, var_Frequent_Invest):

    # cumulation of all investments

    arr_invest = np.zeros(len(logR))
    arr_invest[0] = var_First_Invest
    
    # find new month
    months = logR.index.month
    month_changes = (months[1:] != months[:-1])

    # cummulated investment
    arr_invest[1:][month_changes] = var_Frequent_Invest
    
    return np.cumsum(arr_invest)

def calc_compound_end_value(logR_raw, months_raw, start_inv, cont_inv):

    # the end value is a linear combination of the compounded single values
    # this compounding is calculated using a matrix calculation

    # compound start invest
    compound_start_inv = start_inv * np.exp(np.sum(logR_raw))
    
    # compounded returns
    total_logR = np.sum(logR_raw)

    # compounded returns until end of each perios
    cum_logR = np.cumsum(logR_raw)

    # find new month
    month_changes = np.where(months_raw[1:] != months_raw[:-1])[0] + 1
    
    # the actual compounding until last day is the total minus the remaining growth
    growth_factors = np.exp(total_logR - cum_logR[month_changes - 1])

    # final summation
    final_savings_val = np.sum(cont_inv * growth_factors)
    
    return compound_start_inv + final_savings_val

def calc_risk_metrics(logR):
    
    # calculates longest duration and loss of a drawdown
    # basis are log returns

    cum_returns = logR.cumsum()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns - running_max
    max_drawdown_pct = (np.exp(drawdowns.min().iloc[0]) - 1) * 100
    is_in_drawdown = drawdowns < 0
    max_duration = 0
    current_duration = 0

    for in_drawdown in is_in_drawdown.values:
        if in_drawdown:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0

    max_duration = max(max_duration, current_duration)
            
    return max_drawdown_pct, max_duration

def calc_btd(logR, close_prices, tAxis, start_val, monthly_val, res_pct, dip_limit_dec):
    # convert to np
    logR_raw = logR.values
    prices_raw = close_prices.values
    months = tAxis.month.values
    tAxis_raw = tAxis.values 

    # daily yield
    daily_tg_rate = np.log(1.025) / 252
    tg_factor = np.exp(daily_tg_rate)
    
    # three year rolling windows of drawdowns
    rolling_peak = close_prices.rolling(window=252*3, min_periods=1).max()
    all_drawdowns = ((close_prices / rolling_peak) - 1).values 

    # allocate etf assets from investment reserve (IR)
    val_etf = start_val * (1 - res_pct)
    val_ir = start_val * res_pct
    
    # init df_btd value
    df_btd = np.zeros(len(logR_raw))
    df_btd[0] = val_etf + val_ir
    
    # monitor full investments
    arr_buy_dates, arr_buy_values = [], []

    # find the dip and invest IR
    for i in range(1, len(logR_raw)):
        
        # normal growth
        val_etf *= np.exp(logR_raw[i-1])
        val_ir *= tg_factor
        
        drawdown = all_drawdowns[i]
        
        # check ddip
        if drawdown <= -dip_limit_dec and val_ir > 0:
            val_etf += val_ir
            arr_buy_dates.append(tAxis_raw[i])
            arr_buy_values.append(val_etf)
            val_ir = 0
            
        # check new month and calculate total asset value
        # check recovery and rebalance in new month 
        if months[i] != months[i-1]:
            # investment of new month
            val_etf += monthly_val
            
            # rebalance, if needed
            if drawdown >= -0.005:
                total_val = val_etf + val_ir
                val_ir = total_val * res_pct
                val_etf = total_val * (1 - res_pct)
        
        df_btd[i] = val_etf + val_ir
        
    return df_btd, arr_buy_dates, arr_buy_values

def plot_charts(tAxis, title, df_invest, df_base, df_base_label, df_strategy=None, 
                           df_comp_color=None, df_comp_label=None):
    fig = go.Figure()

    # invested
    fig.add_trace(go.Scatter(x=tAxis, y=df_invest, name="investiert", 
                             line=dict(color='#3498db', width=1.5, dash='dot'),
                             hovertemplate="investiert: %{y:,.0f} €<extra></extra>"))

    # baseline
    fig.add_trace(go.Scatter(x=tAxis, y=df_base, name=df_base_label, 
                             line=dict(color='#2ecc71', width=1.5),
                             hovertemplate=f"{df_base_label}: %{{y:,.0f}} €<extra></extra>"))

    # comparison
    if df_strategy is not None:
        fig.add_trace(go.Scatter(x=tAxis, y=df_strategy, name=df_comp_label, 
                                line=dict(color=df_comp_color, width=1.5),
                                hovertemplate=f"{df_comp_label}: %{{y:,.0f}} €<extra></extra>"))
    
    # layout
    fig.update_layout(
        template="plotly_dark", 
        hovermode="x unified", 
        xaxis_title="Jahr",
        yaxis_title="Wert in EUR",
        yaxis_tickformat=",.0f",
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        separators=",.",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def section_UI_setup():
    # --- UI Setup ---
    st.set_page_config(page_title="FinChamp - Welt-ETF kannst du selbst", layout="wide")
    st.title("Die Kunst des klugen Investierens: Das Weltportfolio")

    st.write("""**Gute Investoren sind gute Risikomanager.** Das Credo dieser Seite ist deshalb so banal wie robust: Privatanleger interessiert, ob sie am Ende wahrscheinlich **mehr oder weniger Geld im Portemonnaie** haben.""")

    st.write(f"""
            Darauf ist diese Seite ausgerichtet. Sie zeigt, warum ein einfacher Welt-ETF solide Rendite bringt und gleichzeitig viele Anlagerisiken inhärent reduziert.

            Außerdem wollen wir eine Lücke schließen: Die üblichen Informationsseiten und Blogs über die ETF-Anlage zeigen weder eine geschlossene Darstellung noch eine interaktive (z.B. wie bei Zinsrechnern).
            Meist wird nur behauptet, ohne sich um eine Nachweisführung zu bemühen. Schon gar nicht wird modernes Risikomanagement adressiert.
            Gleichzeitig taugen akademische Kennzahlen für Privatanwender wiederum auch nicht, weil sie recht unverständlich sind. 
            """)

    st.success("""
               **Fazit vorab:** Wie stellt sich ein kluger Investor auf? Er investiert:
               - **weltweit gestreut**
               - **kostengünstig**
               - **langfristig**
               - **stumpf** (ununterbrochen)
               - **skeptisch** (gegenüber Versprechen, Prospekten und tollen Stories)

               **Lösung: Ein Sparplan in einen Welt-ETF** erfüllt alle diese Kriterien. Der ist auch noch so pflegeleicht, dass man sich auf die Zufuhr von frischem Geld durch höheres Einkommen konzentrieren kann. 
               """)
    st.write("""
            **Kontext:** 
            - Alle Überlegungen gelten für den **Vermögensaufbau**. 
            - Für das Entsparen ändert sich die Sicht. Hier wird eine **Reserve wichtiger** (Gold oder Cash), um Krisen zu überbrücken. In der Ansparphase ist diese meist ein Rendite-Killer (siehe unten).
            - Mit Reserve meinen wir einen Topf zum taktischen Investieren; nicht den privaten Notgroschen, z.B. für die kaputte Waschmaschine. 
            - Alle Angaben verstehen sich vor Steuern und vor Inflation.""")

    st.write("""**Haftungsausschluss:** Historische Daten und Simulationen sind keine Garantie für zukünftige Entwicklungen. 
            Die hier gezeigten Charts und Berechnungen dienen der Bildung und Information, nicht der Anlageberatung. Wir übernehmen keine Haftung für Ihre persönlichen Investmententscheidungen.
    """)
def section_world_analysis(df):
    # --- Section Header ---
    st.subheader("Der langfristige Erfolg mit einem Welt-ETF")

    # --- Section Data ---
    logR = calc_logReturn(df)
    
    df_base = calc_historical_df(logR, var_First_Invest, var_Frequent_Invest)
    tAxis = df.index[1:]
    
    df_invest = calc_invest_df(logR, var_First_Invest, var_Frequent_Invest)

    # --- KPI Calculation ---
    final_value = df_base[-1]
    total_paid = df_invest[-1]
    profit = final_value - total_paid

    daily_mean_return = (np.exp(logR.mean().iloc[0]) - 1) * 100
    daily_win_chance = (len(logR[logR.iloc[:, 0] > 0]) / len(logR)) * 100

    ## --- Drawdown Calculation ---
    max_dd, max_dur = calc_risk_metrics(logR)
    diff_to_investment = np.array(df_base) - np.array(df_invest)
    negative_diffs = diff_to_investment[diff_to_investment < 0]
    if len(negative_diffs) > 0:
        rel_loss = (diff_to_investment / df_invest)
        max_nominal_loss_pct = rel_loss.min() * 100
        duration_below_investment = np.sum(diff_to_investment < 0)
    else:
        max_nominal_loss_pct = 0
        duration_below_investment = 0

    # --- KPI Plot ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Gesamt investiert", f"{format_de(total_paid)} €")

    c2.metric("Endvermögen", f"{format_de(final_value)} €")

    c3.metric("Gewinn/Verlust", f"{format_de(profit)} €", delta=f"{((final_value/total_paid)-1)*100:.2f}%")

    c4.metric("Längste Durststrecke des Index", f"{max_dur} Tage",  delta=f"{max_dd:.1f} %", 
                help="Die maximale Zeit und der theoretische Verlust **des Index** nach einem Crash.")
    
    c5.metric("Größte Kapitalverlust", f"{duration_below_investment} Tage", delta=f"{max_nominal_loss_pct:.1f} %", 
                help="Die maximale Zeit und der theoretische Verlust, wo das Vermögen **unter deine Einzahlungen** gefallen ist")
    
    c6.metric("Tägliche Gewinnchance", f"{daily_win_chance:.1f} %", delta=f"Ø {daily_mean_return:+.2f} % Rendite täglich",
            help=f"Steigt oder fällt unser investiertes Vermögen häufiger?")

    # --- Chart Plot ---
    fig_stock_picking = plot_charts(tAxis, 'Entwicklung Welt-ETF Portfolio', df_invest, 
                                    df_base, 'Welt-ETF')
    
    st.plotly_chart(fig_stock_picking, width='stretch', key="chart_hist")

    # --- Section Conclusion ---
    st.success("""
               **Wir wurden gezinkt!** Aber positiv: Von heute auf morgen Geld zu verdienen, ist etwas wahrscheinlicher als bei einem fairen Münzwurf. 
               Langfristig führt das zu erheblichem Vermgögensaufbau.
               """)

def section_manager_vs_etf(df):
    # --- Section Header ---
    st.write("---")
    st.subheader("Sollten wir auf den Dr. Manager vertrauen?")
    st.write("""
             Fondsverkäufer behaupten gerne, ein promovierter Manager würde mit seinen ausgefeilten Methoden, Algorithmen
              und Hochleistungsrechnern bessere Ergebnisse erzielen. Das koste uns **'nur' in etwa 2% jährlich**. 
             Was bedeutet das für unser Endvermögen?
            """)
    
    # --- Section Data ---
    logR_base = calc_logReturn(df)
    tAxis = df.index[1:]

    daily_costs = np.log(1.02) / 252
    logR_fonds = logR_base - daily_costs

    df_base = calc_historical_df(logR_base, var_First_Invest, var_Frequent_Invest)
    df_fonds = calc_historical_df(logR_fonds, var_First_Invest, var_Frequent_Invest)
    df_invest = calc_invest_df(logR_base, var_First_Invest, var_Frequent_Invest)

    # --- KPI Calculation ---
    total_paid = df_invest[-1]
    final_etf = df_base[-1]
    profit_etf = final_etf - total_paid
    
    final_fonds = df_fonds[-1]
    profit_fonds = final_fonds - total_paid
    
    cost_fonds = final_etf - final_fonds
    cost_fonds_pct = (final_fonds / final_etf - 1) * 100

    # --- KPI Plot ---       
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Gesamt investiert", f"{format_de(total_paid)} €")
    
    c2.metric("Endvermögen ETF", f"{format_de(final_etf)} €", 
                delta=f"{format_de(profit_etf)} € Gewinn")
    
    c3.metric("Endvermögen Fonds", f"{format_de(final_fonds)} €", 
                delta=f"{format_de(profit_fonds)} € Gewinn")
    
    c4.metric("Kosten des Fondsmanagers", f"{format_de(-cost_fonds)} €",
            delta=(f"{cost_fonds_pct:.1f} % geringeres Endvermögen" if cost_fonds_pct < 0
                   else f"{cost_fonds_pct:.1f} % höheres Endvermögen"))

    # --- Chart Plot ---
    fig_stock_picking = plot_charts(tAxis, 'Die Kostenschere: Manager vs. Index-ETF', df_invest, 
                                    df_base, 'Welt-ETF', df_fonds, '#9b59b6', 'Welt-Fonds')
    
    st.plotly_chart(fig_stock_picking, width='stretch', key="chart_fonds")

    # --- Section Conclusion ---
    st.error(f"""
        **Gut zu wissen 'Asset-Allokation':** Um diese Kosten zu kompensieren, müsste der Fondsmanager höhere Gewinne erzielen. 
             Wie will er das anstellen, wenn er auch nur auf weltweite Aktien zugreifen kann? 
                Er müsste geschickt das Geld auf verschiedene Anlageprodukte verteilen. Warum macht das in Zeiten von KI nicht jeder?
                Absicherung kostet Geld - im Beispiel werden uns **{format_de(cost_fonds)} € in Rechnung** gestellt.
                
        Nur **kaufen wir damit im Schnitt weder Sicherheit noch höhere Rendite**. Untersuchungen (SPIVA-Report) zeigen: 
        Über Zeiträume von 15 Jahren schneiden Fondsmanager schlechter ab als der Index.
    """)

    st.warning(f"""
    **Gut zu wissen:** Lass dich nicht von "nur 2 % jährliche Gebühren" täuschen. Das klingt wenig, wird aber sehr teuer. 
               **Rechne Prozentzahlen immer in echte Euro um.** Nur so sieht man die Wahrheit.
    """)

def section_wirecard_analysis(df_welt, df_wdi):
    # --- Section Header ---
    st.write("---")
    st.subheader("Einzelaktie WireCard")

    # --- Section Data ---
    df_welt = df_welt.loc["2017-10-01":].copy()

    df_wdi = df_wdi.iloc[:, 0] if isinstance(df_wdi, pd.DataFrame) else df_wdi
    df_base = df_welt.iloc[:, 0] if isinstance(df_welt, pd.DataFrame) else df_welt
    df_sync = pd.merge(pd.DataFrame({'Close': df_wdi}), pd.DataFrame({'Close': df_base}), 
                        left_index=True, right_index=True, how='inner', suffixes=('_wdi', '_world'))
    
    df_wdi = df_sync[['Close_wdi']].rename(columns={'Close_wdi': 'Close'})
    df_base = df_sync[['Close_world']].rename(columns={'Close_world': 'Close'})

    logR_wdi = calc_logReturn(df_wdi)
    logR_base = calc_logReturn(df_base)
    
    df_invest = calc_invest_df(logR_base, var_First_Invest, var_Frequent_Invest)
    df_base = calc_historical_df(logR_base, var_First_Invest, var_Frequent_Invest)
    df_wdi = calc_historical_df(logR_wdi, var_First_Invest, var_Frequent_Invest)

    # there is noise in the stock data, thus stock clamped at delisting date
    tAxis = df_sync.index[1:]
    df_wdi[df_sync.index[1:] >= pd.Timestamp("2020-06-25")] = 0

    # --- KPI Calculation ---
    # none

    # --- KPI Plot ---
    # none

    # --- Chart Plot --- 
    view = st.radio("Vergleiche die Szenarien:", ["Der heiße Aktientipp", "Win or Lose?"], horizontal=True)

    if view == "Der heiße Aktientipp":
        mask = tAxis <= "2018-09-03"
        tAxis, df_wdi, df_base, df_invest = tAxis[mask], df_wdi[mask], df_base[mask], df_invest[mask]
    else:
        tAxis, df_wdi, df_base, df_invest = tAxis, df_wdi, df_base, df_invest

    fig_stock_picking = plot_charts(tAxis, '100% Welt-ETF vs. 100% WireCard', df_invest, 
                                    df_base, 'Welt-ETF', df_wdi, '#e74c3c', 'Einzelaktie' )
    
    st.plotly_chart(fig_stock_picking, width='stretch',key="chart_stock")
    
    # --- Section Conclusion ---
    if view == "Der heiße Aktientipp":
        st.info("**Frage:** Will man wirklich den Reichtum verpassen, den die Einzelaktie suggeriert?")
    else:
        st.error("**Gut zu wissen**: Wer sich auf die Meinung der Finanzaufsicht(!), Medien, Investoren und FinFluencer verließ, musste zusehen, wie sich sein Geld in leere Versprechen auflöste. Das langweilige Weltportfolio lief unbeirrt weiter.")
        st.success("""
                    **Diversifikation** ist der einzige Gratis-Schutz an der Börse. Die gleichzeitige Pleite aller Welt-Aktien ist sehr, sehr unwahrscheinlich.

                    1. **Schutz vor Unwissenheit:** Da wir nicht wissen können, welche Firma morgen betrügt oder pleitegeht, kaufen wir einfach alle.
                    2. **Der Preis:** Du wirst nie die maximale Rendite einer einzelnen Raketen-Aktie erzielen. 
                    3. **Der Lohn:** Du erhältst die **Rendite des gesamten Weltmarktes** – und schläfst ruhig, während Einzelwetten über Nacht wertlos werden können.
        """)

def section_gold_analysis(df_base, df_gold):
    # --- Section Header ---
    st.write("---")
    st.subheader("Einzeltitel Gold")

    # --- Section Data ---
    df_gold_clean = df_gold.iloc[:, 0] if isinstance(df_gold, pd.DataFrame) else df_gold
    df_base_clean = df_base.iloc[:, 0] if isinstance(df_base, pd.DataFrame) else df_base
    
    df_sync = pd.merge(pd.DataFrame({'Close': df_gold_clean}), pd.DataFrame({'Close': df_base_clean}), 
                        left_index=True, right_index=True, how='inner', suffixes=('_gold', '_world'))
    
    logR_gold_all = calc_logReturn(df_sync[['Close_gold']].rename(columns={'Close_gold': 'Close'}))
    logR_base_all = calc_logReturn(df_sync[['Close_world']].rename(columns={'Close_world': 'Close'}))
    
    tAxis_all = df_sync.index[1:]

    ## --- Time Slices ---
    view = st.radio("Wie sieht es bei Gold aus?", ["Rohrkrepierer", "Trauminvestment", "Langfristig"], horizontal=True)

    if view == "Rohrkrepierer":
        mask = tAxis_all <= "2005-12-31"
    elif view == "Trauminvestment":
        mask = tAxis_all >= "2023-01-01"
    else:
        mask = slice(None) # Wählt alles aus

    tAxis_final = tAxis_all[mask]
    logR_gold_final = logR_gold_all[mask]
    logR_base_final = logR_base_all[mask]

    df_gold_path = calc_historical_df(logR_gold_final, var_First_Invest, var_Frequent_Invest)
    df_base_path = calc_historical_df(logR_base_final, var_First_Invest, var_Frequent_Invest)
    df_invest_path = calc_invest_df(logR_gold_final, var_First_Invest, var_Frequent_Invest)

    # --- KPI Calculation ---
    # none

    # --- KPI Plot ---    
    # none

    # --- Chart Plot ---
    fig_gold = plot_charts(tAxis_final, f'100% Welt-ETF vs. 100% Gold ({view})', df_invest_path, 
                           df_base_path, 'Welt-ETF', df_gold_path, '#f1c40f', 'Gold')
    
    st.plotly_chart(fig_gold, width='stretch', key="chart_gold")

    # --- Section Conclusion ---
    st.info("""
    **Gut zu wissen:** Gold wird als Wertspeicher angesehen, als Versicherung gegen Inflation, Krisen 
    oder politische Fehlsteuerung. Aber: Der Goldwert schwankt wie eine Einzelaktie.
    """)

    st.warning(f"""
    **Totalverlust? Unwahrscheinlich** Seit über 5.000 Jahren setzt die Menschheit auf Gold. Das ist eine ziemlich lange Erfolgsbilanz.
            Viel spricht dafür, dass sich diese Werthaltigkeit fortsetzt. **Aber:** Wert erhalten, heißt nicht steigern. Gold baut nichts, 
               erforscht nichts, entwickelts nichts, zahlt keine Löhne. Es könnte auch wieder ein Rohrkrepierer werden.
    """)

    st.success("""
               Zur **Versicherung vor soziopolitischen Risiken** kann man Gold anteilig ins Anlagevermögen aufnehmen. 
               """)
    
    st.write("""Wir prüfen als nächstes, ob die Mischung eines Welt-ETF mit Gold wesentliche Vorteile bringt.""")

def section_etf_gold_mix(df_base, df_gold):
    # --- Section Header ---
    gold_cost = 0.005
    st.write("---")
    st.subheader("Diversifikation verbessern: Gold ins Welt-Depot!")
    
    st.write("""
        Wie wirkt sich Gold auf unseren Anlageerfolg aus? Wir mischen den **Welt-ETF** mit einem festen Anteil **Gold**. 
        Das Portfolio wird monatlich auf die Zielgewichtung glattgezogen (Rebalancing).
        """)
    st.write(f"**Versicherungskosten**: Versicherungen kosten Geld. Gold hat Lagerkosten oder einen höheren ETC-Spread. Wir setzen diese Kosten mit {gold_cost*100:.1f} % jährlich an.")

    ## --- Slider for Gold Allocation ---
    gold_pct = st.slider("Gold-Anteil im Portfolio", 0, 30, 10, 5, format="%d%%",
                         help="Wie viel Prozent deines Kapitals sollen dauerhaft in Gold gehalten werden? Der Rest fließt in den Welt-ETF.")
    gold_ratio = gold_pct / 100
    etf_ratio = 1.0 - gold_ratio

    # --- Section Data ---
    df_sync = pd.merge(pd.DataFrame({'Close': df_gold.iloc[:,0]}), 
                        pd.DataFrame({'Close': df_base.iloc[:,0]}), 
                        left_index=True, right_index=True, how='inner', suffixes=('_gold', '_world'))
    
    logR_gold = calc_logReturn(df_sync[['Close_gold']].rename(columns={'Close_gold': 'Close'}))
    logR_gold += np.log(1 - gold_cost)/252
    logR_world = calc_logReturn(df_sync[['Close_world']].rename(columns={'Close_world': 'Close'}))
    
    tAxis = df_sync.index[1:]

    logR_portfolio = (logR_gold * gold_ratio) + (logR_world * etf_ratio)

    df_strategy = calc_historical_df(logR_portfolio, var_First_Invest, var_Frequent_Invest)
    df_base = calc_historical_df(logR_world, var_First_Invest, var_Frequent_Invest)
    df_invest = calc_invest_df(logR_world, var_First_Invest, var_Frequent_Invest)

    # --- KPI Calculation ---
    final_mix = df_strategy[-1]
    final_pure = df_base[-1]
    
    diff_euro = final_mix - final_pure
    diff_pct = (final_mix / final_pure - 1) * 100

    # --- KPI Plot ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Anteil Gold", f"{gold_pct} %")
    c2.metric("Endvermögen Mix", f"{format_de(final_mix)} €")
    c3.metric("Endvermögen 100% ETF", f"{format_de(final_pure)} €")
    c4.metric("Unterschied", f"{format_de(diff_euro)} €", delta=f"{diff_pct:.1f} %")

    # --- Chart Plot ---
    fig_mix = plot_charts(tAxis, f'Portfolio-Vergleich: {100-gold_pct}% Welt-ETF / {gold_pct}% Gold', 
                          df_invest, df_base, '100% Welt-ETF', df_strategy, '#f1c40f', 'ETF-Gold-Mix')
    
    st.plotly_chart(fig_mix, width='stretch', key="chart_etf_gold_mix")

    # --- Section Conclusion ---
    st.info(f"""
    **Was die Goldversicherung bewirkt.** Wir erkennen am Chart drei Dinge sehr schön:
            
    1. Gold zum Welt-ETF beigemischt schwächt **Kursrückgänge** ab.

    2. Gold zum Welt-ETF beigemischt schwächt **Kurssteigerungen** ab. 

    3. Eine Versicherung kostet Geld, in diesem Beispiel {format_de(-diff_euro)} €.

    """)
    
    st.warning("""
               **Wie bewerten wir die Diversifikation?**     
               - Wurde unser Vermögen tatsächlich diversifiziert? Bei einem Welt-ETF ist das eine schwierige Frage.  Da ein Totalverlust eines Welt-ETF sehr, sehr unwahrscheinlich ist: Was bringt weitere Diversifikation?
               - Oder schützt Gold vor nicht-finanziellen Risiken: soziopolitische Krisen, Währungskrisen, Tauschgeschäfte?
    """)
    
    return gold_ratio

def section_backtest_gold(df_base, df_gold, gold_ratio):
    # --- Section Header ---
    st.subheader("Risikoanalyse: Wie hätte sich die Goldbeimischung in der echten Welt geschlagen?")
    st.write(f"Wir testen, wie sich ein Portfolio mit **{gold_ratio*100:.0f}% Gold** im Vergleich zum reinen Welt-ETF über hunderte historische Zeiträume geschlagen hat:")

    st.write(f"Anhand der historischen Zeiträume wird berechnet, um **wie viel** (im Median) und **wie oft** 'mit Gold' besser war als 'ohne Gold':")
    
    # --- Section Data ---
    df_sync = pd.merge(pd.DataFrame({'Close': df_gold.iloc[:,0]}), pd.DataFrame({'Close': df_base.iloc[:,0]}), 
                        left_index=True, right_index=True, how='inner', suffixes=('_gold', '_world'))
    
    logR_gold_raw = calc_logReturn(df_sync[['Close_gold']].rename(columns={'Close_gold': 'Close'})).values
    logR_world = calc_logReturn(df_sync[['Close_world']].rename(columns={'Close_world': 'Close'})).values
    
    # Lagerkosten Gold abziehen (0.5% p.a. -> täglicher Abzug)
    logR_gold = logR_gold_raw - np.log(1 - 0.005)/252

    # year slices
    years_to_test = [1, 3, 5, 10, 20]

    # progress bar for all years
    cols = st.columns(len(years_to_test))
    progress_bar = st.progress(0)
    total_steps = sum([len(range(0, len(logR_world) - (y * 252), 21)) for y in years_to_test])
    current_step = 0
    
    # iterate through the slices 
    for idx, y in enumerate(years_to_test):
        window_size = y * 252
        step = 21 # Einmal pro Monat (Börsenmonat) ansetzen
        
        results_diffs = [] 
        start_indices = range(0, len(logR_world) - window_size, step)
        
        if len(start_indices) > 0:
            for start_idx in start_indices:
                end_idx = start_idx + window_size
                
                s_logR_world = logR_world[start_idx:end_idx]
                s_logR_gold = logR_gold[start_idx:end_idx]
                
                # Rebalancing
                s_logR_mix = (s_logR_gold * gold_ratio) + (s_logR_world * (1 - gold_ratio))
                
                # perforamnce difference
                perf_world = np.exp(np.sum(s_logR_world))
                perf_mix = np.exp(np.sum(s_logR_mix))
                
                diff_pct = (perf_mix / perf_world - 1) * 100
                results_diffs.append(diff_pct)

                # progress update 
                current_step += 1
                if current_step % 10 == 0:
                    progress_bar.progress(min(current_step / total_steps, 1.0))
            
            # --- KPI Calc and Plot per year ---
            if results_diffs:
                # --- KPI Calculation ---
                wins = [d for d in results_diffs if d > 0]
                win_rate = (len(wins) / len(results_diffs)) * 100
                median_perf = np.median(results_diffs)
                
                # --- KPI Plot ---
                cols[idx].metric(
                    label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                    value=f"{median_perf:+.1f} %",
                    delta=f"Gold-Anteil war in {win_rate:.0f}% besser",
                    delta_color="normal" if win_rate > 50 else "inverse"
                )
        else:
            cols[idx].write(f"{y}J: Zu wenig Daten")
    
    progress_bar.empty()

    # --- Section Conclusion ---
    st.success(f"""
        **Bestätigt:** Im Schnitt ist Gold eine Versicherung, die uns im Welt-ETF tendenziell Geld kostet. Ob diese Kosten den psychologischen Schutz wert sind, muss jeder selbst entscheiden.""")

def section_btd_analysis(df):
    # --- Section Header ---
    st.write("---")
    st.subheader("Smart Investieren? Günstige Gelegenheiten abpassen")
    
    st.write(f"Viele Ratgeber und FinFluencer suggerieren, man könne durch 'smartes' Abpassen von Kursrückgängen den Anlageerfolg verbessern. Machen wir die Probe aufs Exempel:")
    st.write(f"- **'stumpf'** investieren: Wenn wir Geld übrig haben, wird es direkt investiert, z.B. per Sparplan.")
    st.write(f"- **'smart'** investieren: Wir halten eine Reserve vor, die wir zu günstigen Zeitpunkten investieren, d.h. wenn der Markt um einen gewissen Prozentsatz gefallen ist.")

    ## --- Slider for BTD calibration ---
    col_a, col_b = st.columns(2)

    with col_a:
        reserve_pct = st.slider("Höhe der Reserve", 5, 30, 15, 5, format="%d%%",
                                help="Wir teilen dein gesamtes Kapital (Startbetrag + monatliche Sparrate) in den Welt-ETF und eine Reserve für günstige Gelegenheiten. Die Reserve wird mit 2,5% pro Jahr verzinst. Der Regler legt die Größe der Reserve fest.")
        reserve_pct = reserve_pct / 100
    with col_b:
        dip_limit = st.slider("Kauf-Schwelle für Gelegenheit", 5, 40, 15, 5, format="%d%%",
            help="Ab wie viel Prozent Kurssturz soll die Reserve investiert werden? "
                "Eine Krise wird oft ab -20% definiert. Man schaut dabei auf Preisabstamd zum Hoch der letzten 3 Jahre, nicht auf das Allzeithoch."
        )
        dip_limit_dec = dip_limit / 100
    
    # --- Section Data ---
    close_prices = df.iloc[:, 0]
    logR_df = calc_logReturn(df)
    logR = logR_df.iloc[:, 0]
    tAxis = df.index
    
    df_btd, arr_buy_dates, arr_buy_values = calc_btd(
        logR, close_prices, tAxis, var_First_Invest, var_Frequent_Invest, reserve_pct, dip_limit_dec
    )
    
    df_base = calc_historical_df(logR, var_First_Invest, var_Frequent_Invest)
    
    # --- KPI Calculation ---
    final_100 = df_base[-1]
    final_btd = df_btd[-1]

    df_invest = calc_invest_df(logR, var_First_Invest, var_Frequent_Invest)
    total_paid = df_invest[-1]
    
    profit_100 = final_100 - total_paid
    profit_btd = final_btd - total_paid
    diff_euro = final_btd - final_100
    diff_pct = (final_btd / final_100 - 1) * 100
    diff = df_btd[-1] - df_base[-1]

    # --- KPI Plot ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("Gesamt investiert", f"{format_de(total_paid)} €")
    
    c2.metric("Endvermögen stumpf", f"{format_de(final_100)} €", 
                delta=f"{format_de(profit_100)} € Gewinn")
    
    c3.metric("Endvermögen smart", f"{format_de(final_btd)} €", 
                delta=f"{format_de(profit_btd)} € Gewinn")
    
    c4.metric("Anzahl der Gelegenheiten", f"{len(arr_buy_dates)}")
    
    c5.metric("Kosten des Smart-seins", f"{format_de(diff_euro)} €",
                delta=f"{diff_pct:.1f} % Unterschied", )

    # --- Chart Plot ---
    fig_btd = plot_charts(tAxis, 'Stumpf (Buy and Hold) vs. Smart (günstige Gelegenheiten abpassen)',
                                     df_invest, df_base, 'stumpf', df_btd, '#e67e22', 'smart')
    
    ## --- Mark full invest ---
    if arr_buy_dates:
        fig_btd.add_trace(go.Scatter(
            x=arr_buy_dates, y=arr_buy_values, mode='markers', name='Kauf (Dip)',
            marker=dict(color="#f1660f", size=12, symbol='star', line=dict(width=1, color='white')),
            hovertemplate="Reserve investiert<extra></extra>"
        ))

    st.plotly_chart(fig_btd, width='stretch', key="chart_btd")

    # --- Section Conclusion ---
    if diff < 0:
        st.error(f"""
        **Ergebnis:** Ganz so 'smart' war die Strategie wohl nicht. Die Kosten belaufen sich auf **{format_de(abs(diff))} €**.
        **Wie kommt das?** Durch die Reserve ist ein Teil des Geldes für längere Zeit schlechter verzinst. Ein Welt-ETF ist schlicht zu performant, als dass sich der Aufwand einer Reserve lohnen würde.
        """)
        st.success(f"""
                **'Time in the Market beats Timing the Market:**  Je früher und länger wir investieren, desto besser! **Wenn wir Geld übrig haben: Investieren!**
         """)
        
        st.info(f"""
        **Achtung bei Aussagen über Investmentrenditen:** Meistens werden die Barreserven nicht mitgerechnet, weil die das Ergebnis verschlechtern.
                Für eine korrekte Einschätzung muss man die ETF-Anlage, Barreserven, Immobilien, Anleihen, Gold, Bitcoin, etc. addieren.
        """)

    else:
        st.success(f"**Ergebnis:** Dein Eingriff hat das Ergebnis verbessert. Das ist eine große Ausnahme. Schreib uns doch, wie du das gemacht hast 😊.")

    return reserve_pct, dip_limit_dec  

def section_backtest_btd(df_full, res_pct, dip_limit_dec):
    # --- Section Header ---
    st.write("---")
    st.subheader("Risikoanalyse: Wie schlug sich 'smart' in der echten Welt?")
    st.write(f"Anhand der historischen Zeiträume wird berechnet, um **wie viel** (im Median) und **wie oft** 'smart' besser als 'stumpf' war:")
    
    # --- Section Data ---
    all_prices = df_full.iloc[:, 0].values
    all_dates = df_full.index
    all_months = df_full.index.month.values
    
    logR_all_df = calc_logReturn(df_full)
    all_logR = logR_all_df.iloc[:, 0].values

    # year slices
    years_to_test = [1, 3, 5, 10, 20]

    # progress bar for all years
    cols = st.columns(len(years_to_test))
    progress_bar = st.progress(0)
    total_steps = sum([len(range(0, len(all_logR) - (y * 252), 21)) for y in years_to_test])
    current_step = 0
    
    # iterate through the slices 
    for idx, y in enumerate(years_to_test):
        window_size = y * 252
        step = 21
        
        results_diffs = [] 
        
        start_indices = range(0, len(all_logR) - window_size, step)
        
        if len(start_indices) > 0:
            for start_idx in start_indices:
                end_idx = start_idx + window_size
                
                s_logR = all_logR[start_idx:end_idx]
                s_months = all_months[start_idx:end_idx]
                s_prices = all_prices[start_idx:end_idx]
                s_dates = all_dates[start_idx:end_idx]
                

                df_base = calc_compound_end_value(s_logR, s_months, var_First_Invest, var_Frequent_Invest)
                
                df_smart, _, _ = calc_btd(pd.Series(s_logR), pd.Series(s_prices), s_dates, 
                                          var_First_Invest, var_Frequent_Invest, res_pct, dip_limit_dec)
                
                # perforamnce difference
                diff_pct = (df_smart[-1] / df_base - 1) * 100
                results_diffs.append(diff_pct)

                # progress update 
                current_step += 1
                if current_step % 5 == 0:
                    progress_bar.progress(min(current_step / total_steps, 1.0))
            
            # --- KPI Calc and Plot per year ---
            if results_diffs:
                # --- KPI Calculation ---
                wins = [d for d in results_diffs if d > 0]
                win_rate = (len(wins) / len(results_diffs)) * 100
                median_perf = np.median(results_diffs)
                
                # --- KPI Plot ---
                cols[idx].metric(
                    label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                    value=f"{median_perf:+.1f} %",
                    delta=f"In {win_rate:.0f}% der Zeiträume war 'smart' besser",
                    delta_color="normal" if win_rate > 50 else "inverse"
                )
        else:
            cols[idx].write(f"{y}J: Zu wenig Daten")
    
    progress_bar.empty()

    # --- Chart Plot ---
    # none

    # --- Section Conclusion ---
    st.warning("""
            **Fazit:** Der **Hype um 'smartes' Investieren ist historisch nicht gerechtfertigt.** Die Erfolgschance durch 'smartes' 
               Verhalten lässt sich von einem Münzwurf kaum unterscheiden - und vermutlich ist der Rest statistisches Rauschen.
                            Wenn wir noch bedenken, dass 'smart' mehr Zeitaufwand und höhere Transaktionskosten bedeutet, ist es so gut wie nie besser.
                **Es gibt keine rationale Gründe von 'stumpfem Investieren' abzuweichen.**
             """)

def section_monte_carlo(df, reserve_pct, dip_limit_dec):
    # --- Section Header ---
    n_sims = 500 
    years_to_test = [1, 3, 5, 10, 20]
    st.write("---")
    st.subheader("Risikoanalyse: Wie schlägt sich 'smart' in Paralleluniversen?") 
    st.write(f"Anstatt der historischen Daten erzeugen wir {format_de(n_sims * len(years_to_test))} gänzlich andere Kurse und vergleichen, um **wie viel** (im Median) und **wie oft** 'smart' gegen 'stumpf' gewinnt:")
    
    ## Normal regime
    logR_df = calc_logReturn(df)
    logR_all = logR_df.iloc[:, 0]
    mu_normal = logR_all.mean()
    sigma_normal = logR_all.std()
    
    ## crash regime
    mu_crash = -0.02
    sigma_crash = 0.03
    
    ## crash state machine
    P = np.array([
        [1 - 1/(60*21), 1/(60*21)], # Aus Normal: Bleibe normal / Wechsle zu Crash
        [1/(18*21), 1 - 1/(18*21)]  # Aus Crash: Wechsle zu normal / Bleibe Crash
    ])

    ## prepare KPI plot
    cols = st.columns(len(years_to_test))

    ## progress bar
    progress_bar = st.progress(0)
    total_steps = len(years_to_test) * n_sims
    current_step = 0

    # --- Section Data per time window ---
    for idx, y in enumerate(years_to_test):
        days = 252 * y
        results_diffs = [] # Sammelt die Performance-Unterschiede (%)
        indices = np.arange(0, days, 21) 

        # Regime-Pfade vorab würfeln
        regimen_matrix = np.zeros((days, n_sims), dtype=int)
        current_regimen = np.zeros(n_sims, dtype=int)
        for t in range(1, days):
            probs = P[current_regimen] 
            rand_vals = np.random.rand(n_sims)
            current_regimen = np.where(rand_vals < probs[:, 0], 0, 1)
            regimen_matrix[t, :] = current_regimen

        # simulate paths within time window 
        for i in range(n_sims):
            path_regimes = regimen_matrix[:, i]
            
            mus = np.where(path_regimes == 0, mu_normal, mu_crash)
            sigmas = np.where(path_regimes == 0, sigma_normal, sigma_crash)
            sim_logR_raw = np.random.normal(mus, sigmas)
            
            sim_prices_raw = 100 * np.exp(np.cumsum(sim_logR_raw))
            sim_logR_ser = pd.Series(sim_logR_raw)
            sim_prices_ser = pd.Series(sim_prices_raw)
            sim_dates = pd.date_range(start="2026-01-01", periods=days, freq='B')

            # calc base 
            cum_logR_to_end = sim_logR_raw[::-1].cumsum()[::-1]
            growth_factors = np.exp(cum_logR_to_end[indices])
            df_base = (var_First_Invest * np.exp(sim_logR_raw.sum())) + (var_Frequent_Invest * growth_factors).sum()

            # calc BTD
            df_smart, _, _ = calc_btd(sim_logR_ser, sim_prices_ser, sim_dates, 
                var_First_Invest, var_Frequent_Invest, reserve_pct, dip_limit_dec)
            
            # perforamnce difference and append
            diff_pct = (df_smart[-1] / df_base - 1) * 100
            results_diffs.append(diff_pct)

            # progress update
            current_step += 1
            if current_step % 5 == 0:
                progress_bar.progress(min(current_step / total_steps, 1.0))

        # --- KPI Calc and Plot per year ---
        if results_diffs:
            # --- KPI Calculation ---
            wins = [d for d in results_diffs if d > 0]
            win_rate = (len(wins) / n_sims) * 100
            median_perf = np.median(results_diffs)
            
            # --- KPI Plot ---
            cols[idx].metric(
                label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                value=f"{median_perf:+.1f} %",
                delta=f"In {win_rate:.0f}% der Universen war 'smart' besser",
                delta_color="normal" if win_rate > 50 else "inverse"
            )

    progress_bar.empty()

    # --- Section Conclusion ---
    st.warning("""
        **Fazit** In Paralleluniversen ist es noch klarer: Selbst wenn wir günstige Bedingungen für die 'smarte' Welt schaffen 
        (indem wir langanhaltende Crash-Phasen simulieren), gibt es **keinen rationalen Grund vom 'stumpfen Investieren' abzuweichen.**
        """)
    
def main():

    section_UI_setup()

    try:
        df_welt = get_stock_data(ASSETS["Welt"]["ticker"])
        df_wdi = get_stock_data(ASSETS["WDI"]["ticker"])
        df_gold = get_gold_data(get_stock_data(ASSETS["Gold"]["ticker"]))

        if df_welt.empty or df_wdi.empty or df_gold.empty:
            st.error(f"Konnte nicht alle Daten laden. Bitte Seite neu laden.")
            st.stop()


        section_world_analysis(df_welt)

        c1, c2 = st.columns(2)
        with c1: section_wirecard_analysis(df_welt, df_wdi)

        with c2: section_gold_analysis(df_welt, df_gold)

        gold_ratio = section_etf_gold_mix(df_welt, df_gold)

        section_backtest_gold(df_welt, df_gold, gold_ratio)

        section_manager_vs_etf(df_welt)

        res_pct, dip_lim = section_btd_analysis(df_welt)

        section_backtest_btd(df_welt, res_pct, dip_lim)

        section_monte_carlo(df_welt, res_pct, dip_lim)

    except Exception as e:
        st.error(f"Fehler beim Datenabruf: {e}")
        st.stop()

if __name__ == "__main__":
    main()