import streamlit_authenticator as stauth
print(stauth.Hasher(["your_password"]).generate())