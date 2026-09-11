import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuración visual de la página
st.set_page_config(page_title="Premier League Moneyball", page_icon="⚽", layout="centered")

st.title("⚽ Premier League: Tasador de Mercado IA")
st.markdown("Herramienta interactiva de **Moneyball** para detectar jugadores infravalorados o sobrevalorados con base en su rendimiento real.")

# 1. Cargar datos en caché (descarga rápida)
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/cleaned_players.csv"
    df = pd.read_csv(url)
    
    # Filtrar jugadores activos (mínimo 450 minutos jugados)
    df = df[df['minutes'] >= 450].copy()
    df['valor_millones'] = df['now_cost'] / 10
    df['nombre_completo'] = df['first_name'] + ' ' + df['second_name']
    
    # Entrenar modelo
    features = ['goals_scored', 'assists', 'minutes']
    X = df[features]
    y = df['valor_millones']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    df['valor_predicho'] = modelo.predict(X)
    df['diferencia'] = df['valor_predicho'] - df['valor_millones']
    return df, modelo

df_activos, modelo = cargar_datos()

# 2. Selector de jugador
jugadores_disponibles = sorted(df_activos['nombre_completo'].unique())
jugador_seleccionado = st.selectbox("Selecciona o busca un futbolista:", options=jugadores_disponibles)

# Datos del jugador elegido
datos_j = df_activos[df_activos['nombre_completo'] == jugador_seleccionado].iloc[0]

# 3. Métricas visuales en tarjetas
col1, col2, col3 = st.columns(3)
col1.metric("Precio Real", f"£{datos_j['valor_millones']:.2f}M")
col2.metric("Valor según IA", f"£{datos_j['valor_predicho']:.2f}M")
col3.metric("Diferencia", f"£{datos_j['diferencia']:+.2f}M", delta=f"{datos_j['diferencia']:+.2f}M")

# Veredicto dinámico
dif = datos_j['diferencia']
if dif > 1.0:
    st.success(f"🟢 **Veredicto: GANGA / INFRAVALORADO** (Produce £{dif:.2f}M por encima de su precio)")
elif dif < -1.0:
    st.error(f"🔴 **Veredicto: SOBREVALORADO** (Cuesta £{abs(dif):.2f}M más de lo que produce técnicamente)")
else:
    st.info("⚪ **Veredicto: PRECIO JUSTO DE MERCADO**")

st.markdown(f"**Estadísticas registradas:** {int(datos_j['goals_scored'])} goles | {int(datos_j['assists'])} asistencias | {int(datos_j['minutes'])} minutos.")

# 4. Gráfica comparativa general
st.subheader("Posición en la Liga")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(df_activos['valor_millones'], df_activos['valor_predicho'], alpha=0.3, color='gray', label='Resto de jugadores')
ax.scatter(datos_j['valor_millones'], datos_j['valor_predicho'], color='red', s=120, edgecolors='black', label=jugador_seleccionado)

limite = max(df_activos['valor_millones'].max(), df_activos['valor_predicho'].max()) + 1
ax.plot([4, limite], [4, limite], color='blue', linestyle='--', label='Valor Justo')

ax.set_xlabel('Precio de Mercado Real (£M)')
ax.set_ylabel('Valor Calculado por IA (£M)')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

st.pyplot(fig)
