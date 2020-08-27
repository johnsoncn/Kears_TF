
from keras.models import load_model

from bank.bank_vali import X_test


loaded_model = load_model('bank_validation.h5')

y_predict = loaded_model.predict(X_test)
print(y_predict)
