
import json
import shutil

"""
Llama 3.x token IDs exceed uint16, so np.uint32 is used for token storage.
    The length of each sample is not fixed
    The index needs to record the starting position and length of each sentence
    The starting position of the sentence is stored in np.uint64, 8B,
    The length is stored in np.uint16, 2B, the maximum is 65536
"""
"""
Merge different data sets:
    1. If they are of the same type, just splice the two files together.
    2. If they are of different types, you need to generate an additional file called .dist. This file stores the total number of samples of each type.
    The .dist file is tentatively the List file saved by torch.save.
"""
"""
warmup: https://stackoverflow.com/questions/11832254/understanding-performance-of-numpy-memmap
"""

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
from transformers import AutoTokenizer
from typing import List
import argparse
import multiprocessing
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
import re

"""try-catch comes from Megatron-Deepspeed/tools/preprocess_data.py"""
try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class CSplitter:
    def tokenize(self, text):
        # Split C code at function definitions
        pattern = r'(\w+\s+\w+\s*\([^)]*\)\s*{)'
        return re.split(pattern, text)

class BashSplitter:
    def tokenize(self, text):
        # Split Bash script code at function definitions
        pattern = r'\bfunction\s+(\w+)\s*\{|(\w+)\s*\(\s*\)\s*\{'
        return re.split(pattern, text)

class CSharpSplitter:
    def tokenize(self, text):
        # Split C# code at method definitions
        pattern = r'((?:public|private|protected|internal|static|virtual|override|abstract|sealed|async|\s)+(?:\w+(?:<[^>]+>)?)\s+\w+\s*\([^)]*\)\s*(?:\{|=>))'
        return re.split(pattern, text)

class CPlusPlusSplitter:
    def tokenize(self, text):
        # Split C++ code at function definitions
        pattern = r'((?:(?:\s*inline\s+)?(?:virtual|static|const|auto)?\s*\w+(?:<[^>]*>)?\s+)?\w+\s*\(.*?\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?\{)'
        return re.split(pattern, text)

class GoSplitter:
    def tokenize(self, text):
        # Split Go code at function definitions
        pattern = r'(func\s+(?:\([^)]*\)\s*)?(?:\w+\s*)?(?:\([^)]*\))?\s*(?:\w+\s*)?\{)'
        return re.split(pattern, text)

class JavaSplitter:
    def tokenize(self, text):
        # Split Java code at method definitions
        pattern = r'((?:public|private|protected|static|final|native|synchronized|abstract|transient|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\([^\)]*\)\s*(?:\{?|[^;]))'
        return re.split(pattern, text)

class JavaScriptSplitter:
    def tokenize(self, text):
        # Split JavaScript code at function definitions (including arrow functions)
        pattern = r'((?:async\s+)?(?:function\s*\*?\s*\w*|\(\s*[^)]*\)\s*=>|\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|\w+\s*:\s*(?:async\s*)?function\*?)\s*\(?[^)]*\)?\s*\{)'
        return re.split(pattern, text)

class JuliaSplitter:
    def tokenize(self, text):
        # Split Julia code at function definitions
        pattern = r'(function\s+\w+(?:{[^}]*})?\s*\([^)]*\)|\w+\s*\([^)]*\)\s*=)'
        return re.split(pattern, text)

class PythonSplitter:
    def tokenize(self, text):
        # Split Python code at function and class definitions
        pattern = r'((?:async\s+)?(?:def|class)\s+\w+\s*(?:\([^)]*\))?\s*:)'
        return re.split(pattern, text)

class SQLSplitter:
    def tokenize(self, text):
        # Split SQL code at query definitions
        pattern = r'(?i)(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK|MERGE|EXPLAIN)'
        return re.split(pattern, text)

class RubySplitter:
    def tokenize(self, text):
        # Split Ruby code at method definitions
        pattern = r'((?:def\s+(?:self\.)?|class\s+(?:<<\s*)?)\w+(?:\s*\(.*?\))?\s*(?:\{|do|\n))'
        return re.split(pattern, text)

class RustSplitter:
    def tokenize(self, text):
        # Split Rust code at function definitions
        pattern = r'((?:pub\s+)?(?:async\s+)?fn\s+\w+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*\{)'
        return re.split(pattern, text)

class TypeScriptSplitter:
    def tokenize(self, text):
        # Split TypeScript code at function definitions
        pattern = r'((?:async\s+)?(?:function\s*\*?\s*\w*|\(\s*[^)]*\)\s*=>|\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|\w+\s*:\s*(?:async\s*)?function\*?)\s*\(?[^)]*\)?\s*(?::\s*[^{]+)?\s*\{)'
        return re.split(pattern, text)


"""refer: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/data/indexed_dataset.py#L349"""
def _warmup_mmap_file(path):
    return
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        # assert os.path.isfile(model_path), model_path
        self.sp_model = AutoTokenizer.from_pretrained(model_path, token=hf_token)

        # BOS / EOS token IDs
         # """ Modified to qwen's bos and eos """
        self.n_words: int = self.sp_model.vocab_size
        self.eos_id: int = self.sp_model.eos_token_id
        self.pad_id: int = self.sp_model.pad_token_id
        self.bos_id: None

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str, print(f"type:{type(s)},content:{s}")
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

class DistributedTokenizer:
    def __init__(self, args, eos: bool, bos: bool, collate_fn=None):
        self.args = args
        self.max_seq_length = self.args.seq_length
        self.eos = eos
        self.bos = bos
        self.collate_fn = collate_fn
        # Used to convert json format into text format. For example, sometimes you may need to splice "title" and "content" together.
        # The input of this function is one line of original text (json), and the output is also one line (that is, the document to be processed)

    def split(self, lst: List[int]):
        '''
        The original function will introduce the last block whose length is less than seq_length into the repeated corpus. Here is the modification:
        If this sentence exceeds the maximum length seq_length, discard it
        '''
        maxlen = self.max_seq_length
        merged_lst = []
        i = j = 0
        answer_lst = []
        while i < len(lst):
            ans = [0, 0]
            ans[0] = i
            sums = lst[i]
            j = i + 1
            while j < len(lst) and sums + lst[j] <= maxlen:
                sums += lst[j]
                j += 1
            k = j  # Record the end point (excluding this point)
            ans[1] = j
            i = k  #
            merged_lst.append(sums)
            answer_lst.append(ans)
        if len(merged_lst) >= 2 or sums > maxlen:
            print(f"one exceed max seq_len, sums:{sums}, merged_lst length:{len(merged_lst)}")
            merged_lst=[]
            answer_lst=[]

        return merged_lst,answer_lst  # Close left and open right

    """code from dEEPsPEED mEGAtRON"""

    def dsmt_initializer(self):
        """Load tokenizer"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)

        if self.args.language.lower() in ['bash', 'c', 'c#', 'c++', 'go', 'java', 'javascript', 'julia', 'neo4j database and cypher', 'python', 'relation database and sql', 'ruby', 'rust', 'typescript', 'text']:
            if self.args.do_split_functions:
                if self.args.language.lower() == "bash": DistributedTokenizer.splitter = BashSplitter()
                elif self.args.language.lower() == "c": DistributedTokenizer.splitter = CSplitter()
                elif self.args.language.lower() == "c#": DistributedTokenizer.splitter = CSharpSplitter()
                elif self.args.language.lower() == "c++": DistributedTokenizer.splitter = CPlusPlusSplitter()
                elif self.args.language.lower() == "go": DistributedTokenizer.splitter = GoSplitter()
                elif self.args.language.lower() == "java": DistributedTokenizer.splitter = JavaSplitter()
                elif self.args.language.lower() == "javascript": DistributedTokenizer.splitter = JavaScriptSplitter()
                elif self.args.language.lower() == "julia": DistributedTokenizer.splitter = JuliaSplitter()
                elif self.args.language.lower() == "neo4j database and cypher": DistributedTokenizer.splitter = Neo4jCypherSplitter()
                elif self.args.language.lower() == "python": DistributedTokenizer.splitter = PythonSplitter()
                elif self.args.language.lower() == "relation database and sql": DistributedTokenizer.splitter = SQLSplitter()
                elif self.args.language.lower() == "ruby": DistributedTokenizer.splitter = RubySplitter()
                elif self.args.language.lower() == "rust": DistributedTokenizer.splitter = RustSplitter()
                elif self.args.language.lower() == "typescript": DistributedTokenizer.splitter = TypeScriptSplitter()
                elif self.args.language.lower() == "text": DistributedTokenizer.splitter = IdentitySplitter()
            else:
                DistributedTokenizer.splitter = IdentitySplitter()
        else:
            assert False, "The currently supported languages are 'bash', 'c#', 'c++', 'go', 'java', 'javascript', 'julia', 'python', 'ruby', 'rust', 'typescript', and 'text'. Please make sure you enter them correctly."

    def _re_split(self, src: str, tokenized: List, start_part=False, end_part=False):
        """

        :param src:         original sentence
        :param tokenized:   The list after dividing the words

        :param start_part:  The incoming src is the beginning part, indicating that there is BOS at the beginning of the current tokenized
        :param end_part:    The incoming src is the end part, indicating that the current tokenized end has EOS.
        :return:
        """

        if len(tokenized) <= self.max_seq_length:
            return [tokenized]
        else:
            assert False, "The current sentence exceeds the maximum length"

    def dsmt_encode(self, json_line):
        if self.collate_fn == None:
            text = json_line
        else:
            # The data format here needs to be processed into the corresponding format
            text = self.collate_fn(json_line)

        if text == "\n" or text.strip() == "" or text == r"\n":
            return []
        # """Only enabled when an error occurs"""
        # text = text.encode('utf-8', 'replace').decode('utf-8')
        """split it into sentences"""
        # print(DistributedTokenizer.splitter)
        sentences = DistributedTokenizer.splitter.tokenize(text)
        # The default is Indientity splitter, so the length will be 1, only one sentence
        assert len(sentences) == 1

        """Segment the sentences, and then split them again if they exceed the length. After the processing is completed, send them to split for fusion."""
        if len(sentences) == 1:
            # only one sentence
            _tokenized = DistributedTokenizer.tokenizer.encode(sentences[0], bos=self.bos, eos=self.eos)
            _tokenized = [_tokenized]
        else:
            _tokenized = []
            for idx, sentence in enumerate(sentences):
                cur_tokenized = DistributedTokenizer.tokenizer.encode(sentence, bos=(idx == 0 and self.bos),
                                                                      eos=(idx == len(sentences) - 1) and self.eos)
                # print(cur_tokenized)
                _tokenized.extend(
                    self._re_split(src=sentence, tokenized=cur_tokenized, start_part=(idx == 0) and self.bos,
                                   end_part=(idx == len(sentences) - 1) and self.eos))

        """Record the number of tokens in each sentence after the clause is broken."""
        length_tokenized = [len(_) for _ in _tokenized]

        """get merged index"""
        _,index = self.split(length_tokenized)
        ultra = []
        for pair in index:
            cur = []
            start, end = pair
            for i in range(start, end):
                cur.extend(_tokenized[i])
            ultra.append(cur)
        """
        ultra = [
            [block1 part tokens],
            [block2 part tokens],
            ...
        ]
        """
        return ultra

    def initializer(self):
        """Load tokenizer"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)

    def encode(self, text: str):
        return DistributedTokenizer.tokenizer.encode(text.strip(), self.bos, self.eos)

class MyDataset(Dataset):
    def __init__(self, data_prefix, seq_length, pad_id):
        super(MyDataset, self).__init__()
        """This requires data_prefix to be a complete path, but does not include the suffix"""
        """For example:/llama/our/data"""
        """/llama/our/data.idx will be automatically added as needed"""
        """/llama/our/data.bin will be automatically added as needed."""
        """/llama/our/data.dis will be automatically added as needed."""
        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"
        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None  # The starting position of each sample
        self.index_length = None  # length of each sample
        self._load_index()
        self._load_bin()
        self._load_dis()

    def _load_index(self):
        """The size in bytes occupied by the file"""
        file_size = os.stat(self.idx_file_path).st_size
        """Total number of samples"""
        assert file_size % 10 == 0  # 2B length, 8B start pos
        self.total_sample = file_size // 10
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).tolist()
            self.index_length = np.frombuffer(f.read(self.total_sample * 2), dtype=np.uint16).tolist()
            # print(self.index_length)

    def _load_bin(self):
        """Referenced to Megatron-Deepspeed"""
        _warmup_mmap_file(self.bin_file_path)
        """Loading large files using memory mapping"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint32, mode='r')

    def _load_dis(self):
        """Only valid when there is a mixture of data from multiple categories"""
        self.distributed = torch.load(self.dis_file_path)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        if self.pad_id == 0:
            data = torch.zeros([self.seq_length], dtype=torch.long)
        else:
            data = torch.ones([self.seq_length], dtype=torch.long) * self.pad_id
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if idx + 1 < self.total_sample:
            assert start_idx + length == self.index_start_pos[idx + 1], \
                f"{start_idx + length}!={self.index_start_pos[idx + 1]}, idx={idx}"
        if length > self.seq_length:
            length = self.seq_length
        # data[0:length] = torch.as_tensor(self.bin_buffer[start_idx:start_idx+length].tolist(), dtype=torch.long)
        # return data
        return self.bin_buffer[start_idx:start_idx + length].tolist()

def count_lines(path):
    """Count the number of lines in the input file"""
    print(path)
    with open(path, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(1024 * 1024 * 1024)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1  # Remove this if a wc-like count is needed
    return count

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="write", type=str, help="There are three modes: merge, write and read.")
    parser.add_argument("--seq_length", default=512, type=int, help="maximum length")
    parser.add_argument("--language", default="python", type=str, help="'bash', 'c#', 'c++', 'go', 'java', 'javascript', 'julia', 'python', 'ruby', 'rust', 'typescript', and 'text'")
    parser.add_argument("--do_split_sentences", action="store_true", default=False, help="Whether to divide the document into sentences")
    parser.add_argument("--do_split_functions", action="store_true", default=False, help="Whether to split the code into functions")
    parser.add_argument("--do_keep_newlines", action="store_true", default=False, help="Whether to retain newlines when dividing")
    parser.add_argument("--file_path", default="examples.txt", type=str, help="Source file, each line is a sample")  # write
    parser.add_argument("--num_workers", default=1, type=int, help="Number of parallel processes")  # write
    parser.add_argument("--tokenizer_path", default='LLaMA2-Tokenizer',type=str, help="Tokenizer file")  # write
    parser.add_argument("--save_prefix", default="train", type=str, help="What is it called when saving? The index file will be added with .idx, and the data file will be added with .bin.")  # write
    parser.add_argument("--save_path", default="path_to_save/", type=str, help="The saved location needs to end with/")  # write
    parser.add_argument("--num_per_doc", default=-1, type=int, help="How many samples are retained for each document? If it is -1, it means all are required.")  # write
    parser.add_argument("--read_path_prefix", default="./hello", type=str,
                        help="The file prefix to be read will be automatically completed when reading .idx/.bin/.dis")  # 读取

    parser.add_argument("--merge_path_prefix", default=None, type=str, help="The file prefix that needs to be merged, ['1','2','3']")
    parser.add_argument("--merge_path_type", default=None, type=str, help="If not provided, it will default to the same type of data set. If provided, it will be given in the format [1,1,0]")
    """The following temporarily cancels the automatic merging function. It must be specified, that is, write to a new file."""
    parser.add_argument("--new_path_prefix", default=None, type=str, help="If it is None, it will automatically select the largest one from the above files for merging. If it is not None, it will automatically")
    """save_mode is temporarily unavailable"""
    parser.add_argument("--save_mode", default=1, type=int, help="0-Do not store index files 1-Store index files")

    return parser.parse_args()

# A line read from jsonl
def collate_fn_from_json(json_line: str):
    data = json.loads(json_line)
    total_text = data['content']
    return total_text

def collate_fn_from_text(text: str):
    return text

def write(args):
    """Count file lines"""
    print(f"[{time.ctime()}] Start counting rows")
    count = count_lines(args.file_path)
    print(f"[{time.ctime()}] Row:{count}")
    """Open text file"""
    fin = open(args.file_path, 'r', encoding='utf-8')
    """Create multiple processes"""
    encoder = DistributedTokenizer(args, eos=True, bos=False, collate_fn=collate_fn_from_json)
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.dsmt_initializer)
    """Read from input stream"""
    # encoded_samples = pool.imap(encoder.dsmt_encode, fin, 25)
    encoded_samples = list(
        (tqdm(pool.imap(encoder.dsmt_encode, fin, 25), total=count, desc="Reading progress"))
    )
    print(f"[{time.ctime()}] 读取完毕")
    """Start writing"""
    """starting position:np.uint64: 8B"""
    """length:np.uint16: 2B"""
    """token:np.uint32: 4B"""
    f_bin_out = open(f"{args.save_path}{args.save_prefix}.bin", "wb")
    encoded_samples = list(encoded_samples)
    pbar = tqdm(total=len(encoded_samples))
    start_pos = 0
    start = []
    length = []
    num_samples = 0
    flag = True
    g = torch.Generator()
    g.manual_seed(2023)
    statistic = [0, 0, 0, 0]  # It starts with s and ends with e, nothing, a complete sentence
    for doc in encoded_samples:
        if args.num_per_doc == -1:
            """Means you want them all"""
            idx = list(range(len(doc)))
            if flag:
                print("Sample all documents")
                flag = False
        else:
            if args.num_per_doc <= 2:
                idx = torch.randint(0, len(doc), [args.num_per_doc], generator=g).tolist()
            else:
                idx = [0, -1]
                if len(doc) > 2:
                    idx.extend((torch.randperm(len(doc) - 2, generator=g) + 1).tolist())
                idx = idx[args.num_per_doc]
            if flag:
                print("Document local sampling")
                flag = False
        for i in idx:
            target = doc[i]
            """If there is none or the length is less than 20, then do not add it."""
            if len(target) == 0:
                continue
            num_samples += 1
            if target[0] == 1 and target[-1] == 2:
                statistic[3] += 1
            elif target[0] == 1 and target[-1] != 2:
                statistic[0] += 1
            elif target[0] != -1 and target[-1] == 2:
                statistic[1] += 1
            else:
                statistic[2] += 1
            f_bin_out.write(np.array(target, dtype=np.uint32).tobytes(order='C'))
            length.append(len(target))
            start.append(start_pos)
            start_pos += len(target)
        pbar.update(1)
    """The following line is the total number of data. In fact, I feel that it can be unnecessary. Just divide the obtained index file size by 10."""
    # f_idx_out.write(np.array([len(encoded_samples)], dtype=np.uint64).tobytes(order='C'))
    f_bin_out.close()
    f_idx_out = open(f"{args.save_path}{args.save_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()
    print(num_samples)
    dis = [num_samples]
    torch.save(dis, f"{args.save_path}{args.save_prefix}.dis")
    torch.save(statistic, f"{args.save_path}{args.save_prefix}.tmp")

def write_scratch(args):
    """Open text file"""
    fin = open(args.file_path, 'r', encoding='utf-8')
    """Instantiate tokenizer"""
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    """Create multiple processes"""
    encoder = DistributedTokenizer(args, eos=False, bos=True)
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.initializer)
    """Read from input stream"""
    encoded_samples = pool.imap(encoder.encode, fin, 25)

    """Start writing"""
    """starting position:np.uint64: 8B"""
    """length:np.uint16: 2B"""
    """token:np.uint32: 4B"""
    f_bin_out = open(f"{args.save_path}{args.save_prefix}.bin", "wb")
    encoded_samples = list(encoded_samples)
    pbar = tqdm(total=len(encoded_samples))
    # start_pos = np.array([0], dtype=np.uint64)
    # length = np.array([0], dtype=np.uint16)
    start_pos = 0
    start = []
    length = []
    num_samples = 0
    for target in encoded_samples:
        """If not, then don’t join"""
        if len(target) == 0:
            continue
        num_samples += 1
        f_bin_out.write(np.array(target, dtype=np.uint32).tobytes(order='C'))
        pbar.update(1)
        length.append(len(target))
        start.append(start_pos)
        start_pos += len(target)
    """The following line is the total number of data. In fact, I feel that it can be unnecessary. Just divide the obtained index file size by 10."""
    # f_idx_out.write(np.array([len(encoded_samples)], dtype=np.uint64).tobytes(order='C'))
    f_bin_out.close()
    f_idx_out = open(f"{args.save_path}{args.save_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint16).tobytes(order='C'))
    f_idx_out.close()

    dis = [num_samples]
    torch.save(dis, f"{args.save_path}{args.save_prefix}.dis")

def read(args):
    ds = MyDataset(args.read_path_prefix, seq_length=args.seq_length, pad_id=0)
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    # bos_token = tokenizer.bos_id
    eos_token = tokenizer.eos_id
    # print("BOS token:", bos_token)
    print("EOS token:", eos_token)

    print(f"Length: {len(ds)}")
    for i in range(len(ds)):
        if i == 20:
            break
        # print(f"Clause:{i}", ds[i])
        print(f"Clause:{i}", tokenizer.decode(ds[i]))
    print(f"The distribution is:{ds.distributed}")

def merge(args):
    """Mixing of different types of data in one dataset is not supported"""
    """Only a single type of data is supported for mixing. Formally speaking, the .dis file has only one data."""
    if args.merge_path_prefix == None:
        assert False
    else:
        merge_path_prefix = eval(args.merge_path_prefix)

    """Determine whether they are of the same type"""
    if args.merge_path_type == None:
        print(f"[{time.ctime()}] Merged datasets are of the same type")
        merge_path_type = None
    else:
        merge_path_type = eval(args.merge_path_type)

    """What is the name of the merged file?"""
    # new_file_path_prefix = -1
    if args.new_path_prefix == None:
        # filesize = [os.stat(file+".bin").st_size for file in merge_path_prefix]
        # max_ = max(filesize)
        # for i in range(len(merge_path_prefix)):
        #     if max_ == filesize[i]:
        #         new_file_path_prefix=i
        #         break
        # print(
        #     f"[{time.ctime()}] The merged datasets are of the same type,"
        #     f"Merge files into {merge_path_prefix[new_file_path_prefix]}")
        assert False
    new_path_prefix = args.new_path_prefix

    """Classify files first"""
    if merge_path_type != None:
        """merge_path_type=[0,0,1,1]"""
        """First build a dict for indexing"""
        classifier_prefix = {}
        for idx, types in enumerate(merge_path_type):
            if types not in classifier_prefix:
                classifier_prefix[types] = [merge_path_prefix[idx]]
            else:
                classifier_prefix[types].append(merge_path_prefix[idx])
        """Binary file, just add it directly to the end"""
        new_file_bin = open(new_path_prefix + ".bin", "wb")
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                with open(file_prefix + ".bin", "rb") as f:
                    shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        """Next writeidx file"""
        new_file_idx = open(new_path_prefix + ".idx", "wb")
        index_start_pos = []
        index_length = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                file_size = os.stat(file_prefix + ".idx").st_size
                """Total number of samples"""
                assert file_size % 10 == 0
                total_sample = file_size // 10
                with open(file_prefix + ".idx", "rb") as f:
                    _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                    _index_length = np.frombuffer(f.read(total_sample * 2), dtype=np.uint16).tolist()
                if len(index_start_pos) > 0:
                    """If it is not the first one, you need to add the currently obtained starting position plus the current starting position of the last sample plus the current length of the last sample."""
                    """For example, I have taken the first file, and the starting position of its last sample is 100. The length is 10, then the offset required for the new file is 110"""
                    index_start_pos.extend((_index_start_pos + index_start_pos[-1] + index_length[-1]).tolist())
                else:
                    """It’s the very beginning, so just join directly"""
                    index_start_pos.extend(_index_start_pos)
                index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint16).tobytes(order='C'))
        new_file_idx.close()
        """writedis file"""
        _cur_size = 0
        new_dist = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                data = torch.load(file_prefix + ".dis")
                assert len(data) == 1
                _cur_size += data[0]
            new_dist.append(_cur_size)
            _cur_size = 0
        torch.save(new_dist, new_path_prefix + ".dis")
        # 做最后的check
        assert sum(new_dist) == len(index_start_pos)
    else:
        """Binary file, just add it directly to the end"""
        new_file_bin = open(new_path_prefix + ".bin", "wb")
        for file in merge_path_prefix:
            with open(file + ".bin", "rb") as f:
                shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        """Index files need to be updated"""
        new_file_idx = open(new_path_prefix + ".idx", "wb")
        index_start_pos = []
        index_length = []
        for file in merge_path_prefix:
            file_size = os.stat(file + ".idx").st_size
            """Total number of samples"""
            assert file_size % 10 == 0  # 2B的length，8B的start pos
            total_sample = file_size // 10
            with open(file + ".idx", "rb") as f:
                _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                _index_length = np.frombuffer(f.read(total_sample * 2), dtype=np.uint16).tolist()
            if len(index_start_pos) > 0:
                """If it is not the first one, you need to add the currently obtained starting position plus the current starting position of the last sample plus the current length of the last sample."""
                """For example, I have taken the first file, and the starting position of its last sample is 100. The length is 10, then the offset required for the new file is 110"""
                index_start_pos.extend((_index_start_pos + index_start_pos[-1] + index_length[-1]).tolist())
            else:
                """It’s the very beginning, so just join directly"""
                index_start_pos.extend(_index_start_pos)
            index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint16).tobytes(order='C'))
        new_file_idx.close()
        assert len(index_start_pos) == len(index_length)
        """Finally generate .dis file"""
        torch.save([len(index_start_pos)], new_path_prefix + ".dis")
        # 下面的代码用于check是否合理
        total = 0
        for file in merge_path_prefix:
            data = torch.load(file + ".dis")
            """Mixing of different types of data in one dataset is not supported"""
            assert len(data) == 1
            total += data[0]
        assert total == len(index_start_pos)

if __name__ == '__main__':
    """There is no judgment on the length of the participle, because it is assumed here that each line is a correct and appropriate"""
    """It seems unclear whether to add EOS to the end of the text and BOS to the beginning of the text."""

    """read"""
    args = get_args()
    print(args)
    if args.mode.lower() == "read":
        """  python my_data_preprocessor.py --mode="read" --read_path_prefix="./dataset1"        """
        """  python my_data_preprocessor.py --mode="read" --read_path_prefix="./merge-english"        """
        """  python my_data_preprocessor.py --mode="read" --read_path_prefix="./merge-chinese"        """
        read(args)
    elif args.mode.lower() == "write":
        """  python my_data_preprocessor.py --mode="write" --file_path="./dataset1.txt" --save_prefix="dataset1" --save_path="./"    """
        """  python my_data_preprocessor.py --mode="write" --file_path="./dataset2.txt" --save_prefix="dataset2" --save_path="./"    """
        """  python my_data_preprocessor.py --mode="write" --file_path="./dataset3.txt" --save_prefix="dataset3" --save_path="./"    """
        """  python my_data_preprocessor.py --mode="write" --file_path="./dataset4.txt" --save_prefix="dataset4" --save_path="./"    """
        """  python my_data_preprocessor.py --mode="write" --file_path="./dataset5.txt" --save_prefix="dataset5" --save_path="./"    """
        write(args)
    elif args.mode.lower() == "merge":
        """  python my_data_preprocessor.py --mode="merge" --merge_path_prefix="['./dataset1', './dataset2']" --merge_path_type=None --new_path_prefix="./merge-english"    """
        """  python my_data_preprocessor.py --mode="merge" --merge_path_prefix="['./dataset3', './dataset4', './dataset5']" --merge_path_type=None --new_path_prefix="./merge-chinese"    """
        merge(args)
    else:
        assert False
