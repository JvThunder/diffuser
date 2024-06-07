import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('inverse')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    input_dim=2*observation_dim,
    output_dim=action_dim,
    hidden_dim=args.hidden_dim,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    render=False, # we don't need to render in this script
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

inv_model = model_config()
trainer = trainer_config(inv_model, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch1 = utils.batchify(dataset[0])
batch2 = utils.batchify(dataset[1])
print(batch1.observations)
print(batch2.observations)
print(batch1.observations.shape)
print(batch1.actions.shape)
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)