import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
#データの読み込み
data = pd.read_csv("../datasets/example1.txt")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_, y_train, y_ = train_test_split(X, y,  test_size = 0.4, random_state = 1)
X_cv,  X_test, y_cv, y_test = train_test_split(X_, y_, test_size = 0.5, random_state = 1)

lamda_range = np.array([0.1, 0.01])
steps = len(lamda_range)
degree = 3
train_loss = np.zeros(steps)
cv_loss = np.zeros(steps)
x = np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100,steps))

for i in range(steps):
    lamda_ = lamda_range[i]
    model = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        Ridge(alpha = lamda_, solver = "auto")
    )

    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    train_loss[i] = np.mean((yhat_train - y_train)**2)

    yhat_cv = model.predict(X_cv)
    cv_loss[i] = np.mean((yhat_cv - y_cv)**2)

    y_pred[:, i] = model.predict(x.reshape(-1, 1))

import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,2,figsize=(8,4))
fig.suptitle("Tuning Regularization",fontsize = 12)

#比較表示(予測)
ax[0].set_title("predictions vs data",fontsize = 12)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

ax[0].scatter(X_train, y_train, color = "red", label="train")
ax[0].scatter(X_cv, y_cv, color = "orange", label="cv")

ax[0].set_xlim(ax[0].get_xlim()) #後から追加する線のせいで軸が勝手に変わるのを防ぐ
ax[0].set_ylim(ax[0].get_ylim()) #set(min,max)

for i in (0,1):
    ax[0].plot(x, y_pred[:,i],  lw=0.5, label=f"$\lambda =${lamda_range[i]}")
ax[0].legend()


#比較表示(損失)
ax[1].set_title("error vs regularization",fontsize = 12)
ax[1].plot(lamda_range, train_loss[:], label="train error", color = "blue")
ax[1].plot(lamda_range, cv_loss[:],    label="cv error",    color = "orange")
ax[1].set_xscale('log') #x軸対数に
ax[1].set_ylim(*ax[1].get_ylim())
optimal_reg_idx = np.argmin(cv_loss) #err_cvのmin
opt_x = lamda_range[optimal_reg_idx]
#垂直線(x,長さ,...)
ax[1].vlines(opt_x, *ax[1].get_ylim(), color = "black", lw=1)

ax[1].set_xlabel("regularization (lambda)")
ax[1].set_ylabel("error")
ax[1].legend(loc='upper left')
plt.tight_layout()
plt.show()