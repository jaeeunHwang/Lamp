import os

class PerformanceResultParser:
    def __init__(self):
        self.dirlocation = "./performance/SM75_RTX2060"
        self.outFileName = "output.log"
        self.metrics = [
            "gpu_sim_cycle",
            "gpu_sim_insn",
            "gpu_ipc",
            "gpu_occupancy",
            "gpu_tot_issued_cta",
            "L1D_total_cache_accesses",
            "L1D_total_cache_misses",
            "L1D_total_cache_miss_rate",            
            "gpgpu_n_load_insn",
            "gpgpu_n_store_insn",
            "gpgpu_n_shmem_insn"
        ]
        self.results = {
            "GEMM": {metric: None for metric in self.metrics},
            "GEMM_RELU": {metric: None for metric in self.metrics},
            "SOFTMAX": {metric: None for metric in self.metrics},
            "TOTAL": {metric: 0 for metric in self.metrics}  # Initialize totals to 0
        }
        self.weights = {
            "GEMM": 5,
            "GEMM_RELU": 1,
            "SOFTMAX": 1
        }
        self.getAllFolderNames()
        self.getAllResults()

    def getAllFolderNames(self):
        self.dirNames = [os.path.join(self.dirlocation, name) for name in os.listdir(self.dirlocation) if os.path.isdir(os.path.join(self.dirlocation, name))]

    def getAllResults(self):
        for dirName in self.dirNames:
            logFilePath = os.path.join(dirName, self.outFileName)
            if os.path.exists(logFilePath):
                kernel_name = self.getKernelName(dirName)
                if kernel_name in self.results:
                    self.parseLogFile(logFilePath, kernel_name)

    def getKernelName(self, dirName):
        if 'gemm_fp32' in dirName:
            return 'GEMM'
        elif 'gemm_relu' in dirName:
            return 'GEMM_RELU'
        elif 'softmax' in dirName:
            return 'SOFTMAX'
        return None

    def parseLogFile(self, logFilePath, kernel_name):
        with open(logFilePath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                for metric in self.metrics:
                    if metric in line:
                        value = self.extractLastValue(line)
                        if value is not None:
                            weighted_value = value * self.weights[kernel_name]
                            self.results[kernel_name][metric] = weighted_value
                            self.results["TOTAL"][metric] += weighted_value

    def extractLastValue(self, line):
        try:
            value = line.split('=')[-1].strip()
            if value.endswith('%'):
                value = value[:-1].strip()
            if self.is_float(value):
                return float(value)
            if value.isdigit():
                return int(value)
        except Exception as e:
            print(f"Error parsing line: {line}, error: {e}")
        return None

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def printResults(self):
        print("****** Lamp LLM Performance Result ******")
        print("- GPU Config.: SM75_RTX2060\n")
        for metric in self.metrics:
            print(f"[{metric}]")
            for kernel in self.results:
                if kernel != "TOTAL":
                    value = self.results[kernel][metric]
                    print(f"- {kernel}: {value}")
            total_value = self.results["TOTAL"][metric]
            print(f"- TOTAL: {total_value}\n")

if __name__ == "__main__":
    parser = PerformanceResultParser()
    parser.printResults()

