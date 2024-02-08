from sklearn.linear_model import LogisticRegression

def train_model(features, labels):
    model = LogisticRegression()
    model.fit(features, labels)
    return model


def predict(model, features):
    predictions = model.predict(features)
    return predictions