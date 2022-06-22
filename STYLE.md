### Formatting

1. All code is auto-formatted using `yapf --style=google`.

### Conventions

1. Compute loss across a batch as `mean(batch_losses)` rather than
   `sum(batch_losses)` so that batch size is independent of learning rate.