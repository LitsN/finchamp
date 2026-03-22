"""
Investment Dashboard
Author: Nico Litschke
License: CC BY-NC-SA
Created: 2026
Description: Tool to compare the performance and risk of a world portfolio to stock picking, investment reserve, gold, fonds. 
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import os

# --- local path ---
base_path = os.path.dirname(os.path.abspath(__file__))

# --- Ticker ---
ASSETS = {
    #"Welt":{'ticker': '^GDAXI'},
    "Welt":{'ticker': '^990100-USD-STRD'},
    "WDI":{'ticker': 'WDI.HM'},
    "Gold":{'ticker': 'GC=F'},
    "USDEUR":{'ticker': 'USDEUR=X'}
}

def format_de(n):
    return f"{n:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def sync_widgets(target_key, source_key):

    # syncing of changes in shared variables

    new_val = st.session_state[source_key]
    st.session_state[target_key] = new_val
    
    if "_main" in source_key:
        other_side_key = source_key.replace("_main", "_side")
    else:
        other_side_key = source_key.replace("_side", "_main")
    
    st.session_state[other_side_key] = new_val

@st.cache_data(ttl=3600)
def get_stock_data(ticker, csv_label):

    file_path = os.path.join(base_path, csv_label)

    # Load CSV historical data
    df_csv = pd.read_csv(file_path, sep=';', parse_dates=['Date'])
    if ticker == 'usdeur.csv': df_csv['Close'] = df_csv['Close'].ffill().bfill()
    df_csv.set_index('Date', inplace=True)
    df_csv.index = pd.to_datetime(df_csv.index)
    df_csv['Close'] = pd.to_numeric(df_csv['Close'], errors='coerce')

    # Try t load newest ticker data
    try:
        data = yf.Ticker(ticker)

        df_api = pd.DataFrame() # Initialisierung
        df_api = data.history(interval="1d", period="max", end=dt.datetime.now() - pd.Timedelta('1day'), auto_adjust=True)

        if df_api.empty: raise ValueError("API lieferte keine Daten")
        
        df_api.index = pd.to_datetime(df_api.index)
        df_api.index = df_api.index.tz_localize(None)

        merge_date = df_api.index.min()

        df_csv_pre = df_csv[df_csv.index < merge_date][['Close']].copy()
        df = pd.concat([df_csv_pre, df_api[['Close']]])
        df.sort_index(inplace=True)
        
        # Speichern der neuen Daten als Backup
        # df_api['Close'].to_csv(f"{ticker}_newdata.csv", sep=';')

    except:
        df = df_csv[['Close']].copy()
    
    df = df[df.index >= "1980-01-01"]

    return pd.DataFrame(df['Close'])

def calc_usdeur_df(df_ticker, df_usdeur):
    
    full_index = pd.date_range(start='1980-01-01', end=pd.Timestamp.today(), freq='D')

    df_fx_master = pd.DataFrame(index=full_index)

    df_fx_master = df_fx_master.merge(df_usdeur[['Close']], left_index=True, right_index=True, how='left')

    df_fx_master['Close'] = df_fx_master['Close'].ffill().bfill()

    df = pd.merge(df_ticker[['Close']], df_fx_master[['Close']],
        left_index=True, right_index=True, how='left', suffixes=('_usd', '_fx'))
    
    df['Close'] = df['Close_fx'] * df['Close_usd']

    return pd.DataFrame(df['Close'].copy())

def calc_currency_state():
    if st.session_state.var_currency_mode == 'EUR':
        return '€'
    else:
        return '$'
    
def calc_logReturn(df):
    logR = np.log(df / df.shift(1)).dropna()
    return logR

def calc_merged_df(df_base, df_strategy):
    # 1. Series extrahieren
    s_base = df_base.iloc[:, 0] if isinstance(df_base, pd.DataFrame) else df_base
    s_strategy = df_strategy.iloc[:, 0] if isinstance(df_strategy, pd.DataFrame) else df_strategy
    
    # 2. Merge (Suffixe nur intern, damit Spalten eindeutig sind)
    df_sync = pd.merge( pd.DataFrame({'Close': s_base}), pd.DataFrame({'Close': s_strategy}), 
        left_index=True, right_index=True, how='outer',  suffixes=('_base', '_strategy'))

    # 3. Bereinigen
    df_sync.sort_index(inplace=True)
    df_sync.ffill(inplace=True)
    df_sync.dropna(inplace=True)

    # 4. Log-Returns berechnen
    # Wir greifen einfach über die Position (iloc) zu, dann sind Namen völlig egal
    logR_base = calc_logReturn(df_sync[['Close_base']].rename(columns={'Close_base': 'Close'}))
    logR_strategy = calc_logReturn(df_sync[['Close_strategy']].rename(columns={'Close_strategy': 'Close'}))

    return logR_base, logR_strategy, df_sync.index[1:]

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

def calc_btd(logR, close_prices, tAxis, start_val, monthly_val, res_pct, dip_limit_dec):
    # convert to np
    logR_raw = logR.values
    #prices_raw = close_prices.values
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
                             hovertemplate=f"investiert: %{{y:,.0f}} {calc_currency_state()}<extra></extra>"))

    # baseline
    fig.add_trace(go.Scatter(x=tAxis, y=df_base, name=df_base_label, 
                             line=dict(color='#2ecc71', width=1.5),
                             hovertemplate=f"{df_base_label}: %{{y:,.0f}} {calc_currency_state()}<extra></extra>"))

    # comparison
    if df_strategy is not None:
        fig.add_trace(go.Scatter(x=tAxis, y=df_strategy, name=df_comp_label, 
                                line=dict(color=df_comp_color, width=1.5),
                                hovertemplate=f"{df_comp_label}: %{{y:,.0f}} {calc_currency_state()}<extra></extra>"))
    
    # layout
    fig.update_layout(
        template="plotly_dark", 
        hovermode="x unified", 
        xaxis_title="Jahr",
        yaxis_title=f"Wert in {calc_currency_state()}",
        yaxis_tickformat=",.0f",
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        separators=",.",
        legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.02),
        font=dict(size=12)
    )
    
    return fig

def section_UI_heading():
    # --- Intro Text ---
    
    logo_path = os.path.join(base_path, 'logo.png')
    qr_path = os.path.join(base_path, 'qr_code.png')
    qr_app_path = os.path.join(base_path, 'qr_code_app.png')
    path_favicon = os.path.join(base_path, 'favicon.png')

    st.set_page_config(page_title="FinChamp - Welt-ETF kannst du selbst", page_icon=path_favicon, layout="wide")
    
    with st.sidebar:
        st.image(logo_path, width=200)
        st.header("Investitionen")

        st.number_input(f"Start Investition ({calc_currency_state()})",  key="var_First_Invest_side", 
            min_value=0, step=250,
            on_change=sync_widgets, args=("var_First_Invest", "var_First_Invest_side")
        )

        st.number_input(f"Monatliche Sparrate ({calc_currency_state()})", key="var_Frequent_Invest_side", 
            min_value=0, step=100,
            on_change=sync_widgets, args=("var_Frequent_Invest", "var_Frequent_Invest_side")
        )

        st.slider("Anlagedauer:", 10, 45, 15, 5, format="%d Jahre", key="var_invest_duration_side",
            on_change=sync_widgets, args=("var_invest_duration", "var_invest_duration_side"))

        st.segmented_control("Währung", options=["EUR", "USD"], key="var_currency_mode_side",
            on_change=sync_widgets, args=("var_currency_mode", "var_currency_mode_side"))
        
        st.image(qr_app_path, caption="finchamp.streamlit.app", width=145)
        st.image(qr_path, caption='www.finchamp.de', width=145)
        st.sidebar.write(f"© {dt.date.today().year} FinChamp e.V., CC BY-NC-SA")

    with st.expander("Investitionen - Charts und Berechnungen aktualisieren automatisch ", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(f"Start Investition ({calc_currency_state()})", key="var_First_Invest_main", 
                min_value=0, step=250,
                on_change=sync_widgets, args=("var_First_Invest", "var_First_Invest_main")
            )
            
        with col2:
            st.number_input(f"Monatliche Sparrate ({calc_currency_state()})", key="var_Frequent_Invest_main", 
                min_value=0, step=100, 
                on_change=sync_widgets, args=("var_Frequent_Invest", "var_Frequent_Invest_main"),
            )

        c1, c2 = st.columns (2)
        with c1:
            st.slider("Dauer:", 10, 45, 45, 5, format="%d Jahre", key="var_invest_duration_main",
            on_change=sync_widgets, args=("var_invest_duration", "var_invest_duration_main"))
        
        with c2:
            st.segmented_control("Währung", options=["EUR", "USD"], key="var_currency_mode_main",
                on_change=sync_widgets, args=("var_currency_mode", "var_currency_mode_main"))
            
    
    st.write("""**Gute Investoren sind gute Risikomanager.** Das Credo dieser Seite ist deshalb so banal wie robust: Privatanleger interessiert, 
             ob sie am Ende wahrscheinlich **mehr oder weniger Geld im Portemonnaie** haben.""")
    
    st.write(f"""
            Darauf ist diese Seite ausgerichtet. Sie zeigt, warum ein einfacher Welt-ETF solide Rendite bringt und gleichzeitig viele Anlagerisiken inhärent reduziert.
            """)

    st.write(f"""
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

    st.warning("""
            FinChamp e.V. ist ein gemeinnütziger Verein, der Finanzbildung in Schulen trägt. Wir sind 100% unabängig von der Finanz- und Versicherungsindustrie, verkaufen nichts und kassieren keine Provision.
            
            Wollen Sie mehr erfahren oder uns in Ihre Schulen holen? 
               
            Sie finden uns auf https://www.finchamp.de
               """)
    st.write("""**Haftungsausschluss:** Historische Daten und Simulationen sind keine Garantie für zukünftige Entwicklungen. 
            Die hier gezeigten Charts und Berechnungen dienen der Bildung und Information, nicht der Anlageberatung. Wir übernehmen keine Haftung für Ihre persönlichen Investmententscheidungen.
    """)
def section_world_analysis(df):
    # --- Section Header ---
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.subheader("Der langfristige Erfolg mit einem Welt-ETF")

    # --- Section Data ---
    logR = calc_logReturn(df)
    df_base = calc_historical_df(logR, var_First_Invest, var_Frequent_Invest)
    df_invest = calc_invest_df(logR, var_First_Invest, var_Frequent_Invest)

    tAxis = df.index[1:]
    
    # --- KPI Calculation ---
    final_value = df_base[-1]
    total_paid = df_invest[-1]
    profit = final_value - total_paid

    ## Geometrische Rendite
    daily_mean_return = (np.exp(logR.mean().iloc[0] * 252) - 1) * 100
    daily_win_chance = (1 - len(logR[logR.iloc[:, 0] < 0]) / len(logR)) * 100

    ## --- Drawdown ATH Calculation ---
    cum_logR = logR.cumsum()
    running_max = cum_logR.cummax()
    dd_ath = cum_logR - running_max
    max_dd_ath = (np.exp(dd_ath.min().iloc[0]) - 1) * 100
    is_in_dd_ath = dd_ath < 0
    if is_in_dd_ath.any(axis=None):
        streak_dd_ath = (is_in_dd_ath != is_in_dd_ath.shift()).cumsum()
        max_dd_dur_ath = streak_dd_ath[is_in_dd_ath].value_counts().max()
    else:
        max_dd_dur_ath = 0

    ## ---  Drawdown Invest Calculation ---
    df_comparison = pd.DataFrame({'base': df_base.flatten(), 'invested': df_invest.flatten()}, index=tAxis)
    df_comparison['diff'] = df_comparison['base'] - df_comparison['invested']

    ## Filter to avoid noise
    negative_mask = (df_comparison['base'] / df_comparison['invested'] - 1) < -0.01
    
    if negative_mask.any():
        # Nur die wirklich negativen Zeilen betrachten
        rel_loss = df_comparison['diff'] / df_comparison['invested']
        max_dd_invest = rel_loss.min() * 100
        group_id = (negative_mask != negative_mask.shift()).cumsum()
        max_dd_dur_invest = group_id[negative_mask].value_counts().max()
    else:
        max_dd_invest = 0
        max_dd_dur_invest = 0

    # --- KPI Plot ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Gesamt investiert", f"{format_de(total_paid)} {calc_currency_state()}")

    c2.metric("Endvermögen", f"{format_de(final_value)} {calc_currency_state()}")

    c3.metric("Gewinn/Verlust", f"{format_de(profit)} {calc_currency_state()}", delta=f"{((final_value/total_paid)-1)*100:.2f}%")

    c4.metric("Längste Verlustdauer des Index", f"{max_dd_dur_ath/252:.1f} Jahre",  delta=f"{max_dd_ath:.1f} %", 
                help="Die maximale Zeit und der theoretische Verlust **des Index** nach einem Crash.")
    
    c5.metric("Längste Kapitalverlust", f"{max_dd_dur_invest} Tage", delta=f"{max_dd_invest:.1f} %", 
                help="Die maximale Zeit und der theoretische Verlust, wo das Vermögen **unter deine Einzahlungen** gefallen ist." \
                " Typischerweise ganz am Anfang der Investition in einen Welt-ETF")
    
    c6.metric("Gewinnchance", f"{daily_win_chance:.1f} %", delta=f"Ø {daily_mean_return:+.1f} % Rendite p.a.",
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
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.write("---")
    st.subheader("Sollten wir auf den Dr. Manager vertrauen?")
    st.write("""
             Fondsverkäufer behaupten gerne, ein promovierter Manager würde mit seinen ausgefeilten Methoden, Algorithmen
              und Hochleistungsrechnern bessere Ergebnisse erzielen. Das koste uns **'nur' in etwa 2% jährlich** bei sonst gleicher Anlageperformance? 
             Was bedeutet das für unser Endvermögen?
            """)
    
    ## --- Slider for Fonds Cost ---
    daily_costs = st.slider("Höhere jährliche Kosten des Fonds in Prozentpunkten:", .5, 5.0, 2.0, .25, format="%f%%") / 100
    
    # --- Section Data ---
    logR_base = calc_logReturn(df)
    tAxis = df.index[1:]

    daily_costs = np.log(1 - daily_costs) / 252
    logR_fonds = logR_base + daily_costs

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

    c1.metric("Gesamt investiert", f"{format_de(total_paid)} {calc_currency_state()}")
    
    c2.metric("Endvermögen ETF", f"{format_de(final_etf)} {calc_currency_state()}", 
                delta=f"{format_de(profit_etf)} {calc_currency_state()} Gewinn")
    
    c3.metric("Endvermögen Fonds", f"{format_de(final_fonds)} {calc_currency_state()}", 
                delta=f"{format_de(profit_fonds)} {calc_currency_state()} Gewinn")
    
    c4.metric("Kosten des Fondsmanagers", f"{format_de(-cost_fonds)} {calc_currency_state()}",
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
                Absicherung kostet Geld - im Beispiel werden uns **{format_de(cost_fonds)} {calc_currency_state()} in Rechnung** gestellt.
                
        Nur **kaufen wir damit im Schnitt weder Sicherheit noch höhere Rendite**. Untersuchungen (SPIVA-Report) zeigen: 
        Über Zeiträume von 15 Jahren schneiden Fondsmanager schlechter ab als der Index.
    """)

    st.warning(f"""
    **Gut zu wissen:** Lass dich nicht von "nur 2 % jährliche Gebühren" täuschen. Das klingt wenig, wird aber sehr teuer. 
               **Rechne Prozentzahlen immer in echte Euro um.** Nur so sieht man die Wahrheit.
    """)

def section_wirecard_analysis(df_welt, df_wdi):
    # --- Section Header ---
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.write("---")
    st.subheader("Einzelaktie WireCard")

    # --- Section Data ---

    df_welt = df_welt.loc["2017-10-01":].copy()

    # sync df  and calculate log returns
    logR_base, logR_wdi, tAxis = calc_merged_df(df_welt, df_wdi)
    
    df_invest = calc_invest_df(logR_base, var_First_Invest, var_Frequent_Invest)
    df_base = calc_historical_df(logR_base, var_First_Invest, var_Frequent_Invest)
    df_wdi = calc_historical_df(logR_wdi, var_First_Invest, var_Frequent_Invest)

    # there is noise in the stock data, thus stock clamped at delisting date
    df_wdi[tAxis >= pd.Timestamp("2020-06-25")] = 0

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
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.write("---")
    st.subheader("Einzeltitel Gold")

    # --- Section Data ---

    # sync df  and calculate log returns
    logR_base_all, logR_gold_all, tAxis_all = calc_merged_df(df_base, df_gold)

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

    df_gold = calc_historical_df(logR_gold_final, var_First_Invest, var_Frequent_Invest)
    df_base = calc_historical_df(logR_base_final, var_First_Invest, var_Frequent_Invest)
    df_invest = calc_invest_df(logR_gold_final, var_First_Invest, var_Frequent_Invest)

    # --- KPI Calculation ---
    # none

    # --- KPI Plot ---    
    # none

    # --- Chart Plot ---
    fig_gold = plot_charts(tAxis_final, f'100% Welt-ETF vs. 100% Gold ({view})', df_invest, 
                           df_base, 'Welt-ETF', df_gold, '#f1c40f', 'Gold')
    
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
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
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

    # sync df  and calculate log returns
    logR_world, logR_gold, tAxis = calc_merged_df(df_base, df_gold)
    
    # portfolio log return
    logR_gold += np.log(1 - 0.005)/252
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
    c2.metric("Endvermögen ohne Gold", f"{format_de(final_pure)} {calc_currency_state()}")
    c3.metric("Endvermögen mit Gold", f"{format_de(final_mix)} {calc_currency_state()}")
    c4.metric("Unterschied", f"{format_de(diff_euro)} {calc_currency_state()}", delta=f"{diff_pct:.1f} %")

    # --- Chart Plot ---
    fig_mix = plot_charts(tAxis, f'Portfolio-Vergleich: {100-gold_pct}% Welt-ETF / {gold_pct}% Gold', 
                          df_invest, df_base, '100% Welt-ETF', df_strategy, '#f1c40f', 'ETF-Gold-Mix')
    
    st.plotly_chart(fig_mix, width='stretch', key="chart_etf_gold_mix")

    # --- Section Conclusion ---
    st.info(f"""
    **Was die Goldversicherung bewirkt.** Wir erkennen am Chart drei Dinge sehr schön:
            
    1. Gold zum Welt-ETF beigemischt schwächt **Kursrückgänge** ab.

    2. Gold zum Welt-ETF beigemischt schwächt **Kurssteigerungen** ab. 

    3. Eine Versicherung kostet Geld, in diesem Beispiel {format_de(-diff_euro)} {calc_currency_state()}.

    """)
    
    st.warning("""
               **Wie bewerten wir die Diversifikation?**     
               - Wurde unser Vermögen tatsächlich diversifiziert? Bei einem Welt-ETF ist das eine schwierige Frage.  Da ein Totalverlust eines Welt-ETF sehr, sehr unwahrscheinlich ist: Was bringt weitere Diversifikation?
               - Oder schützt Gold vor nicht-finanziellen Risiken: soziopolitische Krisen, Währungskrisen, Tauschgeschäfte?
    """)
    
    return gold_ratio, gold_cost

def section_backtest_gold(df_base, df_gold, gold_ratio, gold_cost):
    # --- Section Header ---
    st.subheader("Risikoanalyse: Wie hätte sich die Goldbeimischung in der echten Welt geschlagen?")
    st.write(f"Wir testen, wie sich ein Portfolio mit **{gold_ratio*100:.0f}% Gold** im Vergleich zum reinen Welt-ETF über hunderte historische Zeiträume geschlagen hat:")

    st.write(f"Anhand der historischen Zeiträume wird berechnet, um **wie viel** (im Mittel) und **wie oft** 'mit Gold' besser war als 'ohne Gold':")
    
    # --- Section Data ---
    # sync df  and calculate log returns, values for BT
    logR_world, logR_gold, _ = calc_merged_df(df_base, df_gold)
    logR_world = logR_world.values
    logR_gold += np.log(1 - gold_cost)/252
    logR_gold = logR_gold.values
    
    # year slices
    years_to_test = [1, 3, 5, 10, 20]

    # initiate KPI Display
    cols = st.columns(len(years_to_test))
    
    # iterate through the slices 
    for idx, y in enumerate(years_to_test):
        window_size = y * 252
        step = 21
        
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
            
            # --- KPI Calc and Plot per year ---
            if results_diffs:
                # --- KPI Calculation ---
                wins = [d for d in results_diffs if d > 0]
                lose_rate = (1 - len(wins) / len(results_diffs)) * 100
                mean_perf = np.mean(results_diffs)
                
                # --- KPI Plot ---
                cols[idx].metric(
                    label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                    value=f"{mean_perf:+.1f} %",
                    delta=f"Mit Gold war in {lose_rate:.0f} % schlechter",
                    delta_color="inverse"
                )

        else:
            cols[idx].write(f"{y}J: Zu wenig Daten")

    # --- Section Conclusion ---
    st.success(f"""
        **Bestätigt:** Im Schnitt ist Gold eine Versicherung, die uns im Welt-ETF tendenziell Geld kostet. Ob diese Kosten den psychologischen Schutz wert sind, muss jeder selbst entscheiden.""")

def section_btd_analysis(df):
    # --- Section Header ---
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.write("---")
    st.subheader("Smart Investieren? Günstige Gelegenheiten abpassen")
    
    st.write(f"Viele Ratgeber und FinFluencer suggerieren, man könne durch 'smartes' Abpassen von Kursrückgängen den Anlageerfolg verbessern. Machen wir die Probe aufs Exempel:")
    st.write(f"- **'stumpf'** investieren: Wenn wir Geld übrig haben, wird es direkt investiert, z.B. per Sparplan.")
    st.write(f"- **'smart'** investieren: Wir halten eine Reserve (2.5% p.a.) vor, die wir zu günstigen Zeitpunkten investieren, d.h. wenn der Markt um einen gewissen Prozentsatz gefallen ist.")

    ## --- Slider for BTD calibration ---
    col_a, col_b = st.columns(2)

    with col_a:
        reserve_pct = st.slider("Höhe der Reserve", 5, 30, 15, 5, format="%d%%",
                                help="Wir teilen dein gesamtes Kapital (Startbetrag + monatliche Sparrate) in den Welt-ETF und eine Reserve für günstige Gelegenheiten. Die Reserve wird mit 2,5% pro Jahr verzinst. Der Regler legt die Größe der Reserve fest.")
        reserve_pct = reserve_pct / 100
    with col_b:
        dip_limit = st.slider("Kauf-Schwelle für Gelegenheit", 5, 40, 20, 5, format="%d%%",
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
    final_base = df_base[-1]
    final_btd = df_btd[-1]

    df_invest = calc_invest_df(logR, var_First_Invest, var_Frequent_Invest)
    total_paid = df_invest[-1]
    
    profit_base = final_base - total_paid
    profit_btd = final_btd - total_paid
    diff_euro = final_btd - final_base
    diff_pct = (final_btd / final_base - 1) * 100
    diff = final_btd - final_base

    # --- KPI Plot ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("Gesamt investiert", f"{format_de(total_paid)} {calc_currency_state()}")
    
    c2.metric("Endvermögen stumpf", f"{format_de(final_base)} {calc_currency_state()}", 
                delta=f"{format_de(profit_base)} {calc_currency_state()} Gewinn")
    
    c3.metric("Endvermögen smart", f"{format_de(final_btd)} {calc_currency_state()}", 
                delta=f"{format_de(profit_btd)} {calc_currency_state()} Gewinn")
    
    c4.metric("Anzahl der Gelegenheiten", f"{len(arr_buy_dates)}")
    
    if diff_euro <= 0:
        c5.metric("Kosten des Smart-seins", f"{format_de(diff_euro)} {calc_currency_state()}",
                delta=f"{diff_pct:.1f} % Unterschied", )
    else:
        c5.metric("Gewinn des Smart-seins", f"{format_de(diff_euro)} {calc_currency_state()}",
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
        **Ergebnis:** Du hast schlechter abgeschnitten.
        Wie kommt das? Die Reserve wurde über längere Zeit schlechter verzinst. Aber gilt das generell? Starte die Risikoanalyse, um es rauszufinden.
        """)

    else:
        st.warning(f"""
                  **Ergebnis:** Du hast besser abgeschnitten. 
                   Wie kommt das? Im Crash hast du bei gleichem Sparbetrag mehr Stück gekauft. Aber gilt das immer? Starte die Risikoanalyse, um es rauszufinden.
        """)
    st.success(f"""
                **'Time in the Market beats Timing the Market:**  Je früher und länger wir investieren, desto besser! **Wenn wir Geld übrig haben: Investieren!**
         """)
    
    st.info(f"""
        **Achtung bei Aussagen über Investmentrenditen:** Meistens werden die Barreserven nicht mitgerechnet, weil die das Ergebnis verschlechtern.
                Für eine korrekte Einschätzung muss man die ETF-Anlage, Barreserven, Immobilien, Anleihen, Gold, Bitcoin, etc. addieren. Und auch in die eigene Währung umrechnen.
        """)

    return reserve_pct, dip_limit_dec  

def section_backtest_btd(df_full, res_pct, dip_limit_dec):
    # --- Section Header ---
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest
    st.write("---")
    st.subheader("Risikoanalyse: Wie schlug sich 'smart' in der echten Welt?")
    st.write(f"Anhand der historischen Zeiträume wird berechnet, um **wie viel** (im Mittel) und **wie oft** 'smart' besser als 'stumpf' war:")
    
    # --- Section Data ---
    all_prices = df_full.iloc[:, 0].values
    all_dates = df_full.index
    all_months = df_full.index.month.values
    
    logR_all_df = calc_logReturn(df_full)
    all_logR = logR_all_df.iloc[:, 0].values

    # year slices
    years_to_test = [1, 3, 5, 10, 20]

    # initiate KPI display
    cols = st.columns(len(years_to_test))
    
    # iterate through the slices 
    with st.spinner("Analysiere historische Zeiträume ..."):
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
                    

                    df_base_final = calc_compound_end_value(s_logR, s_months, var_First_Invest, var_Frequent_Invest)
                    
                    df_smart, _, _ = calc_btd(pd.Series(s_logR), pd.Series(s_prices), s_dates, 
                                            var_First_Invest, var_Frequent_Invest, res_pct, dip_limit_dec)
                    
                    # perforamnce difference
                    diff_pct = (df_smart[-1] / df_base_final - 1) * 100
                    results_diffs.append(diff_pct)
                
                # --- KPI Calc and Plot per year ---
                if results_diffs:
                    # --- KPI Calculation ---
                    wins = [d for d in results_diffs if d > 0]
                    lose_rate = (1- len(wins) / len(results_diffs)) * 100
                    mean_perf = np.mean(results_diffs)
                    
                    # --- KPI Plot ---
                    cols[idx].metric(
                        label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                        value=f"{mean_perf:+.1f} %",
                        delta=f"In {lose_rate:.0f}% der Zeiträume schlechter!",
                        delta_color="inverse"
                    )
            else:
                cols[idx].write(f"{y}J: Zu wenig Daten")

    # --- Chart Plot ---
    # none

    # --- Section Conclusion ---
    st.warning("""
            **Fazit:** Der **Hype um 'smartes' Investieren ist historisch kaum gerechtfertigt.** Die Erfolgschance durch 'smartes' 
               Verhalten lässt sich von einem Münzwurf kaum unterscheiden - und vermutlich ist der Rest statistisches Rauschen.
                            Wenn wir noch bedenken, dass 'smart' mehr Zeitaufwand und höhere Transaktionskosten bedeutet, ist es so gut wie nie besser.
                **Es gibt keine rationale Gründe von 'stumpfem Investieren' abzuweichen.**
             """)

def section_monte_carlo(df, reserve_pct, dip_limit_dec):
    # --- Section Header ---
    n_sims = 250 
    years_to_test = [1, 3, 5, 10, 20]
    var_First_Invest = st.session_state.var_First_Invest
    var_Frequent_Invest = st.session_state.var_Frequent_Invest

    st.write("---")
    st.subheader("Risikoanalyse: Wie schlägt sich 'smart' in Paralleluniversen?") 
    st.write(f"Anstatt der historischen Daten erzeugen wir {format_de(n_sims * len(years_to_test))} gänzlich andere Kurse und vergleichen, um **wie viel** (im Mittel) und **wie oft** 'smart' gegen 'stumpf' gewinnt:")
    
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

    # --- Section Data per time window ---
    with st.spinner("Simuliere Paralleluniversen ..."):
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

            # --- KPI Calc and Plot per year ---
            if results_diffs:
                # --- KPI Calculation ---
                wins = [d for d in results_diffs if d > 0]
                lose_rate = (1- (len(wins) / n_sims)) * 100
                mean_perf = np.mean(results_diffs)
                
                # --- KPI Plot ---
                cols[idx].metric(
                    label=f"{y} {'Jahr' if y==1 else 'Jahre'}", 
                    value=f"{mean_perf:+.1f} %",
                    delta=f"In {lose_rate:.0f}% der Fälle schlechter" if mean_perf <= 0 else f"Nur in {lose_rate-1:.0f}% der Fälle besser!",
                    delta_color="inverse"
                )

    # --- Section Conclusion ---
    st.warning("""
        **Fazit** In Paralleluniversen ist es noch klarer: Selbst wenn wir günstige Bedingungen für die 'smarte' Welt schaffen 
        (indem wir langanhaltende Crash-Phasen simulieren), gibt es **keinen rationalen Grund vom 'stumpfen Investieren' abzuweichen.**
        """)
    
def section_faq():
    with st.expander("FAQ - Häufig gestellte Fragen", expanded=False):
        st.markdown("""
        - **Werde ich mit diesem Ansatz reich?**      
        Ja. Und nein. Das ist eine Frage des Zeithorizonts und des Verhaltens. Kurz- und mittelfristig stehen die Chancen sehr schlecht. Langfristig bei konsequenter
                    Ausführung ist es eine mathematische Notwendigkeit. Die Sicht des Autors: Investieren kostet Zeit. Man kann die bessere Anlage suchen 
                    oder relativ stumpf investieren. Beides gleichzeitig macht man halbherzig. Deshalb: Vollgas bei Qualifikation, Problemlösungskompetenzen und Selbstständigkeit.
                    Das dadurch verdiente Geld schmeißt der Autor über automatisierte Sparpläne in einen Welt-ETF. 

        - **Werde ich durch diesen Ansatz finanziell unabhängig? Baue ich passives Einkommen auf?**      
        Ja. Und nein. Siehe zuvor.
                    
        - **Wie wird der Welt-ETF abgebildet?**      
        Über den *MSCI World Index* (Net Total Return) anstelle eines spezifischen ETFs, da dieser die längste verfügbare Datenhistorie für langfristige Backtests bietet.

        - **Enthält der MSCI World Index (Net Total Return) auch Dividenden?**           
        Ja. Kursveränderungen, Dividende (nach Quellensteuern), Splits.
                    
        - **Ist der MSCI World Index nicht zu US-lastig?**      
        Heute dominieren US-Aktien. Der Index bildet etwa die 1.300 bis 1.600 weltweit größten Aktienunternehmen ab. Sobald US-Aktien schwächeln, rücken andere nach.
                    
        - **Besteht bei einem Welt.ETF Totalverlustrisiko?**      
        Nur theoretisch. Gemäß kybernetischer Systemtheorie ist ein Weltportfolio 'ultrastabil'. Das bedeutet schlicht: Wenn die Kurse stark fallen, kann man ein Schnäppchen schlagen. Das veranlasst Käufer zum Investieren. Das stabilisiert die Kurse und löst erneute Steigerungen aus. Außerdem würden in solchen Krisen Zentralbanken und Regierungen wieder intervenieren.

        - **Stock Picking - Warum keine Krypto, Einzelaktien, Themen-ETF?**      
        Jede Einschränkung, ob nach Regionen, Branchen oder Anzahl der Titel, ist mehr oder minder willkürlich. Diese Auswahl erfordert eine Mischung aus hohem Spezialwissen und hellserischen Fähigkeiten. Das Risiko steigt (siehe das Beispiel *Wirecard*). Ein Welt-ETF reduziert das durch Diversifikation. Wer andere Titel prüfen möchte, kann den Code von github branchen und die Tickersymbole ändern.

        - **Sind Environmental, Social, Governance (ESG) sicherer?**      
        Nein. Siehe Stock Picking.
                    
        - **Sind andere Indexe sicherer?**      
        Nein. Siehe Stock Picking.                    
                    
        - **Werden Inflation und Steuern berücksichtigt?**      
        Nein. Steuersätze und Inflation sind zu individuell, um sie hier valide abbilden zu können.
                    
        - **Sind Wechselkurse eingerechnet?**      
        Ja. Bis 1998 für USD/DM (in EUR konvertiert) mit den historischen Wechselkursen der Bundesbank, danach USD/EUR von Yahoo Finance.
                    
        - **Was sind andere Begriffe für das Vorhalten einer Reserve?**      
        In der Finanzwelt und an der Börse gibt es viele Namen für diese Taktik: *Buy the Dip*, *Investitionsreserve (IR)*, *Gegen den Trend investieren*, *günstige Gelegenheiten abpassen*, *Regression zur Mitte*, *antizyklisches* oder *konträres Investieren*, *Bottom Fishing* oder das Ausnutzen von *Marktkorrekturen*. Bekannte Börsenweisheiten nennen es auch *Räumungsverkauf*, *Schnäppchen kaufen*, *Kaufen, wenn die Kanonen donnern* oder *Kaufen, wenn Blut auf den Straßen fließt*.

        - **Wieso wurde Cost Averaging nicht als eigene Taktik analysiert?**      
        Das *Cost Averaging* (Durchschnittskosteneffekt durch Phaseninvestment) ist letztlich eine Sonderform: Eine Cash-Reserve wird dabei zeitlich gestreckt in Tranchen investiert, anstatt auf ein spezifisches Signal zu warten. Da es sich mathematisch um das schrittweise Auflösen einer Reserve handelt, deckt unsere Analyse zum „Buy the Dip“ die Erfolgswahrscheinlichkeiten dieses Prinzips bereits mit ab. **Wichtig:** Bei einem Sparplan werden neu erwirtschaftete Ersparnisse investiert. Bei der Reserve wird bereits vorhandene Ersparnisse nicht investiert. Das erzeugt Opportunitätskosten.
                    
        - **Warum wird Cash statt Anleihen als Reserve genutzt?**      
        Für Privatanleger ist eine Barreserve (Tagesgeld) oft praktischer zu handhaben, bietet sofortige Liquidität und bietet derzeit ähnliche Zinsen wie AAA Staatsanleihen.

        - **Wieso gilt die Analyse primär für den Vermögensaufbau?**      
        Beim Verkaufen (z.B. Rente) gelten andere Regeln für das Risikomanagement. Verkauft man in einer Krise, kann das Kapital zu schnell aufgebraucht werden. Eine Reserve muss das puffern.

        - **Woher kommen die Daten?**      
        Die historischen Kurse werden automatisiert über die `yfinance` API von Yahoo Finance bezogen. Als Fallback nehmen wir CSV Daten.
                    
        - **Wieso sind die Kursdaten veraltet?**
        Beim Laden wird versucht die neuesten Daten abzurufen. Gelingt das nicht, wird ein Backup verwendet. Diese Backup wird vierteljährlich aktualisiert. Die Analyseergebnisse beeinflusst das kaum.

        - **Sind die Berechnungen verlässlich?**      
        Der Code wurde sorgfältig geprüft und getestet. Da man Fehlerfreiheit nicht beweisen kann, laden wir dich ein, die Berechnungen in unserem Repository zu prüfen und Feedback zu geben: [GitHub von Nico](https://github.com/LitsN/finchamp/)
        
        - **Warum braucht es Backtest und wie funktoniert das?**      
        Würden wir nur die "letzten 10 Jahre" betrachten, wäre die Aussage irreführend. Man muss den Zeitraum nur leicht verschieben und der Anlageerfolg ändert sich massiv. Deshalb nutzen wir "Rolling Windows": Wir schieben ein 10-Jahres-Zeitfenster durch die gesamte Historie und prüfen, was jeweils rauskam. Nur so lässt sich prüfen, ob taktisches Verhalten besser oder schlechter abschneidet. 

        - **Warum ändern sich die Simulationsergebnisse bei jedem Durchlauf?**      
        Das ist gewollt. Wir nutzen so genannte Markov-Ketten und diskretisierte geometrische Brownsche Bewegung für die Monte-Carlo-Simulation. So generieren wir 'unendlich' viele zufällige aber ähnliche Kursverläufe.
                    
        - **Garantieren historische Analysen und Simulationen zukünftigen Anlageerfolg?**      
        Nein. Wir haben auf dieser Seite die Wahrscheinlichkeit aufgezeigt, warum man mit einem kostengünstigen Welt-ETF wenig falsch machen kann.
        """)

    st.success("""
            Was ist wirklich **'smart'? Ein langweiliger Welt-ETF.** Das ist **intellektuell einfach**.
            Aber das **Umsetzen - vor allem diszipliniert Durchalten - ist schwer**.
            Zu viele Versuchungen begegnen uns täglich. Medien und FinFluencer verzerren die Fakten.
            Verkäufer wollen uns 'smartere' Dinge aufdrängen. In der Krise blickt man ängstlich ins Depot.
            Aber die **langfristige Erfolgswahrscheinlichkeit mit einem Welt-ETF steht zu unseren Gunsten**.
            Das ist Fakt und konnte hoffentlich mit dieser Seite vermittelt werden.
               """)
def main():

    st.title("Die Kunst des klugen Investierens: Das Weltportfolio")

    if 'var_First_Invest' not in st.session_state:
        st.session_state['var_First_Invest'] = 1000
        st.session_state['var_First_Invest_side'] = 1000
        st.session_state['var_First_Invest_main'] = 1000

    if 'var_Frequent_Invest' not in st.session_state:
        st.session_state['var_Frequent_Invest'] = 50
        st.session_state['var_Frequent_Invest_side'] = 50
        st.session_state['var_Frequent_Invest_main'] = 50

    if 'var_currency_mode' not in st.session_state:
        st.session_state['var_currency_mode'] = 'EUR'
        st.session_state['var_currency_mode_side'] = 'EUR'
        st.session_state['var_currency_mode_main'] = 'EUR'

    if 'var_invest_duration' not in st.session_state:
        st.session_state['var_invest_duration'] = 15
        st.session_state['var_invest_duration_side'] = 15
        st.session_state['var_invest_duration_main'] = 15
    
    df_welt = get_stock_data(ASSETS["Welt"]["ticker"], 'world_historical.csv')
    df_wdi = get_stock_data(ASSETS["WDI"]["ticker"], 'wdi_historical.csv')     
    df_gold = get_stock_data(ASSETS["Gold"]["ticker"], 'gold_historical.csv')
    df_usdeur = get_stock_data(ASSETS["USDEUR"]["ticker"], 'usdeur.csv')

    if df_welt.empty or df_wdi.empty or df_gold.empty or df_usdeur.empty:
        st.error(f"Konnte nicht alle Daten laden. Bitte Seite neu laden.")
        st.stop()

    else:
        if st.session_state.var_currency_mode == 'EUR':
            df_welt = calc_usdeur_df(df_welt, df_usdeur)
            df_gold = calc_usdeur_df(df_gold, df_usdeur)

        invest_duration = df_welt.index.max() - pd.DateOffset(years=st.session_state.var_invest_duration)

        section_world_analysis(df_welt[df_welt.index >= invest_duration])

        section_UI_heading()

        c1, c2 = st.columns(2)
        with c1: section_wirecard_analysis(df_welt, df_wdi)

        with c2: section_gold_analysis(df_welt, df_gold)

        gold_ratio, gold_cost = section_etf_gold_mix(df_welt[df_welt.index >= invest_duration], df_gold[df_gold.index >= invest_duration])

        section_backtest_gold(df_welt, df_gold, gold_ratio, gold_cost)

        section_manager_vs_etf(df_welt[df_welt.index >= invest_duration])

        res_pct, dip_lim = section_btd_analysis(df_welt[df_welt.index >= invest_duration])

        with st. expander("Risikoanalyse: Backtest und Simulation", expanded=True):
            st.write("Moderne Risikoanalyse der 'smarten' Taktiken. Diese Simulation ist rechenintensiv und benötigt einen kurzen Moment für die Kalkulation.")
            
            if st.button("Risikoanalysen starten"): 
                section_backtest_btd(df_welt, res_pct, dip_lim)
                section_monte_carlo(df_welt, res_pct, dip_lim)

        section_faq()

        first_update = df_welt.index.min().strftime('%d.%m.%Y')
        last_update = df_welt.index.max().strftime('%d.%m.%Y')
        st.sidebar.caption(f"Daten von: {first_update} - {last_update}")

if __name__ == "__main__":
    main()