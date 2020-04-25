import chainer


class ScheduledBatchsizeIncrement(chainer.training.Extension):
    def __init__(self, schedule):
        self.schedule = schedule
   
    def __call__(self, trainer):
        _iterator = trainer.updater._iterators['main']
        current_batch_size = _iterator.batch_size
        next_batch_size = self.schedule.get(trainer.updater._iterators['main'].epoch, current_batch_size)
        if current_batch_size != next_batch_size:
            print('Change batchsize from {} to {}'.format(current_batch_size, next_batch_size))
            iterator_class = _iterator.__class__
            current_epoch = _iterator.epoch
            _iterator._comm.terminate()
            _iterator._prefetch_loop.terminate()
            trainer.updater._iterators['main'] = iterator_class(
                dataset=_iterator.dataset,
                batch_size=next_batch_size,
                repeat=_iterator.repeat, shuffle=_iterator.shuffle,
                n_processes=None, n_prefetch=_iterator.n_prefetch, shared_mem=_iterator.shared_mem,
                order_sampler=_iterator.order_sampler, dataset_timeout=_iterator.dataset_timeout
            )
            trainer.updater._iterators['main'].epoch = current_epoch
            trainer.updater._iterators['main']._set_prefetch_state()


class AdaptiveBatchsizeIncrement(chainer.training.Extension):
    def __init__(self, patience=5, increase_factor=4, maxsize=1000, key='validation/main/loss'):
        self.key = key
        self.val = 99999
        self.patience = patience
        self.stagnated_epoch_count = 0
        self.increase_factor = increase_factor
        self.maxsize = maxsize
   
    def __call__(self, trainer):
        current_val = trainer.observation[self.key]
        if current_val < self.val:
            self.stagnated_epoch_count = 0
            self.val = current_val
            return

        self.stagnated_epoch_count += 1
        if self.stagnated_epoch_count < self.patience:
            return

        _iterator = trainer.updater._iterators['main']
        iterator_class = _iterator.__class__
        current_epoch = _iterator.epoch
        next_batch_size = _iterator.batch_size * self.increase_factor
        if next_batch_size > self.maxsize:
            return True
        _iterator._comm.terminate()
        _iterator._prefetch_loop.terminate()
        print('Change batchsize from {} to {}'.format(_iterator.batch_size, next_batch_size))
        trainer.updater._iterators['main'] = iterator_class(
            dataset=_iterator.dataset,
            batch_size=next_batch_size,
            repeat=_iterator.repeat, shuffle=_iterator.shuffle,
            n_processes=None, n_prefetch=_iterator.n_prefetch, shared_mem=_iterator.shared_mem,
            order_sampler=_iterator.order_sampler, dataset_timeout=_iterator.dataset_timeout
        )
        trainer.updater._iterators['main'].epoch = current_epoch
        trainer.updater._iterators['main']._set_prefetch_state()
        self.stagnated_epoch_count = 0
