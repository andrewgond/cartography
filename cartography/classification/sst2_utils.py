from transformers.data.processors.glue import Sst2Processor

class AdaptedSst2Processor(Sst2Processor):
  def get_examples(self, data_file, set_type):
      return self._create_examples(self._read_tsv(data_file), set_type=set_type)
