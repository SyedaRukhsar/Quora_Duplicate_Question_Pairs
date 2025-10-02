import streamlit as st
import helper

st.header("Quora Duplicate Question Detector")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Find"):
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter both questions")
    else:
        result = helper.predict_pair(q1, q2, thresh=0.5)
        prob = float(result['probability'])   # ensure float
        is_dup = result['is_duplicate']

        st.write(f"**Probability:** {prob:.4f}")

        if is_dup:
            st.success("🟢 These questions are Duplicate ✅")
        else:
            st.info("🔵 These questions are Not Duplicate ❌")
