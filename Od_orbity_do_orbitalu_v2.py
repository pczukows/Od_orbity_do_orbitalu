import hashlib

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Od orbity do orbitalu", page_icon="⚛️", layout="wide")

ORBITALS = {
    "1s": (1, 0, "s", "symetria sferyczna; brak węzłów"),
    "2s": (2, 0, "s", "jeden węzeł radialny"),
    "2p(x)": (2, 1, "px", "płaszczyzna węzłowa yz"),
    "2p(y)": (2, 1, "py", "płaszczyzna węzłowa xz"),
    "2p(z)": (2, 1, "pz", "płaszczyzna węzłowa xy"),
    "3s": (3, 0, "s", "dwa węzły radialne"),
    "3p(x)": (3, 1, "px", "jeden węzeł radialny i jeden kątowy"),
    "3p(y)": (3, 1, "py", "jeden węzeł radialny i jeden kątowy"),
    "3p(z)": (3, 1, "pz", "jeden węzeł radialny i jeden kątowy"),
    "3d(xy)": (3, 2, "dxy", "dwie płaszczyzny węzłowe"),
    "3d(xz)": (3, 2, "dxz", "dwie płaszczyzny węzłowe"),
    "3d(yz)": (3, 2, "dyz", "dwie płaszczyzny węzłowe"),
    "3d(x²−y²)": (3, 2, "dx2y2", "dwie płaszczyzny węzłowe"),
    "3d(z²)": (3, 2, "dz2", "dwie stożkowe powierzchnie węzłowe"),
}
EXTENT = {1: 5.5, 2: 12.0, 3: 20.0}
HELP = {
    "orbital": "Orbital jest stanem jednoelektronowym opisanym funkcją falową ψ. Lista obejmuje rzeczywiste orbitale wodoropodobne s, p i d.",
    "view": "Chmura pokazuje symulowane wyniki pomiarów położenia. Izopowierzchnia pokazuje umowną granicę stałej wartości ψ albo |ψ|². Można pokazać oba elementy.",
    "points": "Liczba punktów oznacza liczbę symulowanych wyników pomiaru dla tak samo przygotowanego stanu. Nie jest liczbą elektronów.",
    "coverage": "Parametr wizualny sterujący rozmiarem powierzchni. Większa wartość daje bardziej rozległą powierzchnię. Nie jest dokładnym, analitycznie scałkowanym procentem prawdopodobieństwa.",
    "phase": "Po zaznaczeniu niebieski i pomarańczowy oznaczają przeciwny znak funkcji ψ. Po wyłączeniu pokazywana jest nieujemna gęstość |ψ|² jednym kolorem.",
    "n": "Główna liczba kwantowa: wpływa na energię i skalę przestrzenną orbitalu.",
    "l": "Poboczna liczba kwantowa: określa liczbę węzłów kątowych i typ orbitalu: l=0 → s, l=1 → p, l=2 → d.",
    "radial": "Węzeł radialny występuje przy określonej odległości r od jądra i ma postać sferycznej powierzchni, na której ψ=0.",
    "angular": "Węzeł kątowy wynika z części kątowej funkcji falowej. Może mieć postać płaszczyzny albo powierzchni stożkowej.",
}


def psi_real(name, x, y, z):
    n, _, kind, _ = ORBITALS[name]
    r = np.sqrt(x*x + y*y + z*z)
    if name == "1s": return np.exp(-r)
    if name == "2s": return (2-r)*np.exp(-r/2)
    if n == 2: return {"px":x,"py":y,"pz":z}[kind]*np.exp(-r/2)
    if name == "3s": return (27-18*r+2*r*r)*np.exp(-r/3)
    if n == 3 and kind in ("px","py","pz"):
        return {"px":x,"py":y,"pz":z}[kind]*(6-r)*np.exp(-r/3)
    angular = {"dxy":x*y,"dxz":x*z,"dyz":y*z,"dx2y2":x*x-y*y,"dz2":2*z*z-x*x-y*y}[kind]
    return angular*np.exp(-r/3)


def radial_function(name, r):
    if name == "1s": return np.exp(-r)
    if name == "2s": return (2-r)*np.exp(-r/2)
    if name.startswith("2p"): return r*np.exp(-r/2)
    if name == "3s": return (27-18*r+2*r*r)*np.exp(-r/3)
    if name.startswith("3p"): return r*(6-r)*np.exp(-r/3)
    return r*r*np.exp(-r/3)


@st.cache_data(show_spinner=False)
def grid_data(name, grid_n):
    extent = EXTENT[ORBITALS[name][0]]
    axis = np.linspace(-extent, extent, grid_n)
    x,y,z = np.meshgrid(axis,axis,axis,indexing="ij")
    psi = psi_real(name,x,y,z)
    return x.ravel(),y.ravel(),z.ravel(),psi.ravel()


def stable_seed(name):
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8],16)


@st.cache_data(show_spinner=False)
def sample_cloud(name,count):
    extent=EXTENT[ORBITALS[name][0]]
    rng=np.random.default_rng(stable_seed(name))
    axis=np.linspace(-extent,extent,62)
    x,y,z=np.meshgrid(axis,axis,axis,indexing="ij")
    psi=psi_real(name,x,y,z).ravel(); weights=psi*psi; weights/=weights.sum()
    idx=rng.choice(len(weights),size=count,p=weights)
    step=2*extent/(len(axis)-1)
    return (x.ravel()[idx]+rng.normal(0,.40*step,count),y.ravel()[idx]+rng.normal(0,.40*step,count),z.ravel()[idx]+rng.normal(0,.40*step,count),np.sign(psi[idx]))


def add_isosurfaces(fig,name,coverage,phase,opacity):
    x,y,z,psi=grid_data(name,36)
    if phase:
        threshold=np.quantile(np.abs(psi),1-coverage/100)
        fig.add_trace(go.Isosurface(x=x,y=y,z=z,value=psi,isomin=threshold,isomax=psi.max(),surface_count=1,colorscale=[[0,"#38bdf8"],[1,"#38bdf8"]],showscale=False,caps=dict(x_show=False,y_show=False,z_show=False),opacity=opacity,name="ψ > 0"))
        if psi.min() < -threshold:
            fig.add_trace(go.Isosurface(x=x,y=y,z=z,value=psi,isomin=psi.min(),isomax=-threshold,surface_count=1,colorscale=[[0,"#f97316"],[1,"#f97316"]],showscale=False,caps=dict(x_show=False,y_show=False,z_show=False),opacity=opacity,name="ψ < 0"))
    else:
        density=psi*psi; threshold=np.quantile(density,1-coverage/100)
        fig.add_trace(go.Isosurface(x=x,y=y,z=z,value=density,isomin=threshold,isomax=density.max(),surface_count=1,colorscale=[[0,"#60a5fa"],[1,"#60a5fa"]],showscale=False,caps=dict(x_show=False,y_show=False,z_show=False),opacity=opacity,name="|ψ|²"))


def orbital_3d_figure(name,view_mode,points,coverage,phase,height=680):
    extent=EXTENT[ORBITALS[name][0]]; fig=go.Figure()
    if view_mode in ("Chmura pomiarów","Chmura + izopowierzchnia"):
        x,y,z,signs=sample_cloud(name,points)
        color=np.where(signs>=0,"#38bdf8","#f97316") if phase else "#7dd3fc"
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",marker=dict(size=2.3,color=color,opacity=.40),name="wyniki pomiarów",hoverinfo="skip"))
    if view_mode in ("Izopowierzchnia","Chmura + izopowierzchnia"):
        add_isosurfaces(fig,name,coverage,phase,.55 if view_mode=="Izopowierzchnia" else .34)
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",marker=dict(size=4,color="#facc15"),name="jądro"))
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=42,b=0),title=f"Model kwantowy — {name}",paper_bgcolor="#07111f",font=dict(color="#e5e7eb"),legend=dict(bgcolor="rgba(0,0,0,.25)"),scene=dict(bgcolor="#07111f",aspectmode="cube",xaxis=dict(title="x/a₀",range=[-extent,extent],gridcolor="#334155"),yaxis=dict(title="y/a₀",range=[-extent,extent],gridcolor="#334155"),zaxis=dict(title="z/a₀",range=[-extent,extent],gridcolor="#334155"),camera=dict(eye=dict(x=1.45,y=1.45,z=1.15))))
    return fig


def bohr_figure(n):
    theta=np.linspace(0,2*np.pi,360); fig=go.Figure()
    for shell in range(1,n+1):
        fig.add_trace(go.Scatter(x=shell*np.cos(theta),y=shell*np.sin(theta),mode="lines",line=dict(color="#60a5fa" if shell==n else "#334155",width=3 if shell==n else 1.5),name=f"n={shell}"))
    fig.add_trace(go.Scatter(x=[n],y=[0],mode="markers",marker=dict(size=13,color="#38bdf8"),name="elektron"))
    fig.add_trace(go.Scatter(x=[0],y=[0],mode="markers",marker=dict(size=4,color="#facc15"),name="jądro"))
    lim=n+.5
    fig.update_layout(height=560,title=f"Model Bohra — poziom n={n}",xaxis=dict(visible=False,scaleanchor="y",range=[-lim,lim]),yaxis=dict(visible=False,range=[-lim,lim]),paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),margin=dict(l=10,r=10,t=40,b=10))
    return fig


def intro(title,text,goals):
    st.header(title); st.info(text); st.markdown("**Po pracy z panelem student powinien:**"); st.markdown("\n".join(f"- {g}" for g in goals))


st.title("⚛️ Od orbity do orbitalu")
st.caption("Interaktywny przewodnik od historycznej trajektorii Bohra do funkcji falowej, gęstości prawdopodobieństwa, fazy i węzłów orbitali wodoropodobnych.")
st.warning("Orbital nie jest orbitą ani powierzchnią, po której porusza się elektron. Chmura punktów przedstawia wyniki wielu pomiarów dla tak samo przygotowanego stanu, a nie wiele elektronów.")

with st.sidebar:
    st.header("Wybór stanu",help="Wszystkie zakładki korzystają z tego samego orbitalu i zestawu parametrów.")
    orbital=st.selectbox("Orbital",list(ORBITALS),help=HELP["orbital"])
    n,l,kind,node_text=ORBITALS[orbital]
    st.caption(f"Wybrano n={n}, l={l}. {node_text}.")
    st.header("Widok 3D",help="Ustawienia są wspólne dla zakładek 1 i 2.")
    view_mode=st.radio("Sposób przedstawienia",["Chmura pomiarów","Izopowierzchnia","Chmura + izopowierzchnia"],help=HELP["view"])
    if view_mode!="Izopowierzchnia":
        points=st.slider("Liczba symulowanych pomiarów",500,12000,4500,500,help=HELP["points"])
    else: points=4500
    if view_mode!="Chmura pomiarów":
        coverage=st.slider("Umowny rozmiar izopowierzchni [%]",70,98,90,help=HELP["coverage"])
    else: coverage=90
    phase=st.checkbox("Kolorem pokaż znak/fazę ψ",True,help=HELP["phase"])
    if phase: st.info("Niebieski oznacza ψ>0, pomarańczowy ψ<0. Kolory nie oznaczają ładunków.")
    else: st.info("Wyłączono fazę: chmura i powierzchnia pokazują nieujemną gęstość |ψ|² jednym kolorem.")

intro("Jak pracować z aplikacją?","Najpierw porównaj klasyczną orbitę z chmurą pomiarów, następnie badaj kształt, znak ψ i węzły. Na końcu połącz obraz 3D z funkcją radialną R(r) i prawdopodobieństwem w sferycznej powłoce.",["odróżniać orbitę od orbitalu","interpretować ψ i |ψ|²","rozpoznawać węzły radialne i kątowe","łączyć liczby n i l z kształtem orbitalu"])

metrics=st.columns(5)
metrics[0].metric("Orbital",orbital,help=HELP["orbital"])
metrics[1].metric("n",n,help=HELP["n"])
metrics[2].metric("l",l,help=HELP["l"])
metrics[3].metric("Węzły radialne",n-l-1,help=HELP["radial"]+" Liczba węzłów radialnych wynosi n−l−1.")
metrics[4].metric("Węzły kątowe",l,help=HELP["angular"]+" Ich liczba wynosi l.")

tab1,tab2,tab3=st.tabs(["1. Orbita ≠ orbital","2. Orbital 3D","3. Część radialna"])

with tab1:
    intro("Ten sam stan w dwóch sposobach opisu","Lewy panel jest historyczną orbitą Bohra zależną tylko od n. Prawy panel pokazuje współczesny rozkład dla pełnego orbitalu, więc rozróżnia s, p i d.",["wskazywać element klasyczny i kwantowy","wyjaśniać, dlaczego Bohr nie rozróżnia 2s i 2p","interpretować punkty jako wyniki pomiarów"])
    st.caption(f"Prawy widok używa aktualnego trybu „{view_mode}” oraz dokładnie tych samych ustawień co zakładka 2.")
    left,right=st.columns(2)
    with left:
        st.plotly_chart(bohr_figure(n),use_container_width=True,config={"displaylogo":False},key=f"bohr-{orbital}")
        st.warning("Punkt elektronu i okrąg są elementami modelu Bohra. Model reaguje na n, lecz nie odróżnia orbitalu s, p ani d w tej samej powłoce.")
    with right:
        st.plotly_chart(orbital_3d_figure(orbital,view_mode,points,coverage,phase,560),use_container_width=True,config={"displaylogo":False},key=f"compare-{orbital}-{view_mode}-{points}-{coverage}-{phase}")
        details=[]
        if view_mode!="Izopowierzchnia": details.append(f"{points:,} wyników pomiarów")
        if view_mode!="Chmura pomiarów": details.append(f"rozmiar izopowierzchni {coverage}%")
        details.append("pokazany znak ψ" if phase else "pokazana gęstość |ψ|²")
        st.success("Aktualny widok: "+", ".join(details)+".")
    st.markdown("### Jak interpretować porównanie?")
    st.markdown("- **Bohr:** elektron ma określoną trajektorię i promień.\n- **Mechanika kwantowa:** stan opisuje ψ; wynik pojedynczego pomiaru położenia jest punktem, ale przed pomiarem nie przypisujemy klasycznej trajektorii.\n- **Chmura:** zagęszczenie punktów odzwierciedla |ψ|².\n- **Izopowierzchnia:** jest umownym narzędziem wizualizacji, nie granicą atomu.")
    st.markdown("### Sprawdź rozumienie")
    q=st.radio("Dlaczego orbitale 2s i 2p wyglądają tak samo w panelu Bohra?",["Mają taki sam ładunek","Model Bohra w tej aplikacji rozróżnia tylko n","Orbital p jest zawsze kulisty"],index=None,key="quiz_compare")
    if q: st.success("Poprawnie: model Bohra przypisuje jeden promień powłoce n=2." if q=="Model Bohra w tej aplikacji rozróżnia tylko n" else "Spróbuj ponownie: model Bohra nie zawiera pełnej informacji o l i części kątowej ψ.")

with tab2:
    intro(f"Szczegółowy widok orbitalu {orbital}","Obracaj model myszą, przybliżaj i porównuj widoki. Te same parametry, które wybrano w panelu bocznym, działają identycznie w zakładce porównawczej.",["rozpoznawać symetrię orbitalu","odczytywać znak/fazę ψ","liczyć i lokalizować węzły"])
    st.plotly_chart(orbital_3d_figure(orbital,view_mode,points,coverage,phase,720),use_container_width=True,config={"displaylogo":False},key=f"detail-{orbital}-{view_mode}-{points}-{coverage}-{phase}")
    c1,c2,c3=st.columns(3)
    c1.info(f"**Węzły:** {node_text}")
    c2.info(f"**Liczba wszystkich węzłów:** n−1 = {n-1}")
    c3.info("**Faza:** znak ψ wpływa na interferencję, ale |ψ|² pozostaje nieujemne.")
    st.markdown("### Co zmienia każda kontrolka?")
    st.markdown("- **Orbital** zmienia n, l, część radialną i kątową ψ.\n- **Sposób przedstawienia** wybiera punkty, powierzchnię albo oba elementy.\n- **Liczba pomiarów** zwiększa gęstość chmury, nie liczbę elektronów.\n- **Rozmiar izopowierzchni** zmienia umowną granicę prezentacji.\n- **Znak/faza ψ** przełącza między dwoma znakami ψ a jednobarwną gęstością |ψ|².")
    st.markdown("### Ćwiczenie kierowane")
    st.markdown("1. Porównaj 1s, 2s i 3s. Policz sferyczne węzły radialne.\n2. Porównaj 2p(x), 2p(y), 2p(z). Wskaż płaszczyznę węzłową.\n3. Porównaj pięć orbitali 3d.\n4. Wyłącz fazę i wyjaśnij, jaka informacja zniknęła.")

with tab3:
    intro(f"Część radialna i prawdopodobieństwo radialne — {orbital}","Fioletowa krzywa przedstawia R(r) ze znakiem. Zielona krzywa uwzględnia powierzchnię sferycznej powłoki i przedstawia prawdopodobieństwo znalezienia elektronu w przedziale odległości.",["odróżniać R(r) od r²|R(r)|²","rozpoznawać węzły radialne","wyjaśniać, dlaczego R może być ujemne, a prawdopodobieństwo nie"])
    rmax={1:12,2:32,3:60}[n]; r=np.linspace(0,rmax,1400); R=radial_function(orbital,r)
    Rn=R/max(np.max(np.abs(R)),1e-15); probability=r*r*R*R; probability/=max(probability.max(),1e-15)
    fig=go.Figure();fig.add_hline(y=0,line_color="#94a3b8",line_width=1)
    fig.add_trace(go.Scatter(x=r,y=Rn,mode="lines",line=dict(color="#a78bfa",width=4),name="R(r) — znak i węzły"))
    fig.add_trace(go.Scatter(x=r,y=probability,mode="lines",line=dict(color="#22c55e",width=4),name="r²|R(r)|² — prawdopodobieństwo w powłoce"))
    fig.update_layout(height=540,xaxis_title="Odległość od jądra r/a₀",yaxis_title="wartość znormalizowana",paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),legend=dict(orientation="h",y=1.08),margin=dict(l=20,r=20,t=35,b=20))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False},key=f"radial-{orbital}")
    st.markdown("### Objaśnienie wykresu")
    st.markdown(r"- $R(r)$ może zmieniać znak. Przejście przez zero poza początkiem wskazuje węzeł radialny.\n- $r^2|R(r)|^2$ jest nieujemne i uwzględnia wzrost powierzchni sferycznych powłok.\n- Maksimum gęstości w punkcie i najbardziej prawdopodobna odległość nie muszą występować w tym samym miejscu.\n- Znak $R(r)$ nie jest znakiem ładunku.")
    st.markdown("### Sprawdź rozumienie")
    q=st.radio("Która krzywa może przyjmować wartości ujemne?",["R(r)","r²|R(r)|²","obie"],index=None,key="quiz_radial")
    if q: st.success("Poprawnie: R(r) zachowuje informację o znaku/fazie." if q=="R(r)" else "Prawdopodobieństwo jest zawsze nieujemne; znak zachowuje R(r).")

st.markdown("---")
st.header("Słownik i wzory")
cols=st.columns(3)
cols[0].info("**Liczby kwantowe**\n\nn=1,2,...\n\nl=0,1,...,n−1\n\nTypy: l=0 → s, l=1 → p, l=2 → d")
cols[1].info("**Węzły**\n\nwszystkie: n−1\n\nradialne: n−l−1\n\nkątowe: l")
cols[2].info("**Prawdopodobieństwo**\n\ngęstość: |ψ|²\n\nradialne: P(r)∝r²|R(r)|²")

st.header("Granice modelu")
st.warning("Aplikacja przedstawia orbitale wodoropodobne i rzeczywiste kombinacje p oraz d. Izopowierzchnie są wyznaczane numerycznie według progu wizualnego. Dla atomów wieloelektronowych funkcje i energie są zmienione przez ekranowanie oraz oddziaływania elektron–elektron.")
st.info("Następny krok: obsadzanie orbitali, spin, zakaz Pauliego, reguła Hunda, konfiguracje elektronowe oraz związek z układem okresowym.")

st.header("Proponowana ścieżka samodzielnej pracy")
st.markdown("1. Porównaj orbitę Bohra i chmurę 1s.\n2. Zwiększ liczbę pomiarów i obserwuj stabilizację chmury.\n3. Porównaj 1s, 2s i 3s.\n4. Obróć 2p(x), 2p(y), 2p(z).\n5. Włącz i wyłącz fazę.\n6. Porównaj pięć orbitali 3d.\n7. Połącz węzły na wykresie R(r) z obrazem 3D.")
