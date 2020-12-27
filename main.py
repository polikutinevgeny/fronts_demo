import streamlit as st
from datetime import date
from utils import load_model
from plot import plot
from data import get_date
from keras import backend as K


@st.cache
def load_date(dt):
    return get_date(dt)


@st.cache(allow_output_mutation=True)
def load_fronts_model():
    model = load_model("weights.hdf5")
    # noinspection PyProtectedMember
    model._make_predict_function()
    model.summary()
    session = K.get_session()
    return model, session


if __name__ == '__main__':
    st.title("Распознавание атмосферных фронтов")
    with st.spinner("Загрузка модели"):
        model, session = load_fronts_model()
    K.set_session(session)
    dt = st.date_input("Дата:", date(2020, 1, 1))
    with st.spinner("Данные атмосферы загружаются сразу за год (в папку data). Это около 2 гигабайт, и может занять "
                    "некоторое время (до 10 минут)."):
        try:
            data = load_date(dt)
        except:
            data = None
            st.text("Нет данных атмосферы на указанную дату")
    if data is not None:
        fronts = model.predict(data).argmax(axis=-1)
        plotted_fronts = plot(data[0], fronts[0], (256, 256), dt)
        st.pyplot(plotted_fronts)


