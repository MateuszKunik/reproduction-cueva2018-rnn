def mean_squared_error(targets, predictions):
    return ((targets - predictions) ** 2).mean()


def l2_weight_regularization(input_weights, output_weights):
    return (input_weights ** 2).mean() + (output_weights ** 2).mean()


def firing_rate_regularization(activities):
    return (activities ** 2).mean()


def compute_loss(
        targets, predictions, in_weights, out_weights, activities, l_w, l_fr,
):
    prediction_loss = mean_squared_error(targets, predictions)
    w_regularization = l2_weight_regularization(in_weights, out_weights)
    fr_regularization = firing_rate_regularization(activities)

    loss = (
        prediction_loss + 
        l_w * w_regularization + 
        l_fr * fr_regularization
    )

    components = {
        "total_loss": loss,
        "prediction_loss": prediction_loss,
        "l2_weight_regularization": w_regularization,
        "firing_rate_regularization": fr_regularization,
    }
    return loss, components


class LinearLambdaScheduler:
    def __init__(self, lambda_min, lambda_max, num_epochs):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_epochs = num_epochs

    def lambda_at(self, epoch: int) -> float:
        if self.num_epochs <= 1:
            return self.lambda_min

        alpha = (epoch - 1) / (self.num_epochs - 1)
        return self.lambda_min + alpha * (self.lambda_max - self.lambda_min)
    

class AdaptiveLambdaScheduler:
    def __init__(self, ratio_min, ratio_max, num_epochs, eps = 1e-8):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.num_epochs = num_epochs
        self.eps = eps

    def ratio_at(self, epoch):
        if self.num_epochs <= 1:
            return self.ratio_min
        
        alpha = (epoch - 1) / (self.num_epochs - 1)
        return self.ratio_min + alpha * (self.ratio_max - self.ratio_min)
    
    def lambda_at(self, epoch, prediction_loss, regularization_loss):
        ratio = self.ratio_at(epoch)

        if regularization_loss < self.eps:
            return 0.0
        
        return ratio * prediction_loss / (regularization_loss + self.eps)