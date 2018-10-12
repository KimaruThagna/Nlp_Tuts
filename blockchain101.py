import datetime
import hashlib
# OOP implementation of a block
class Block:
    blockNo = 0 #default genesis block
    data = None
    next = None # pointer to the next block
    hash = None # block's hash (ID)
    nonce = 0
    previous_hash = 0x0 # previous block
    timestamp = datetime.datetime.now() # synchorizing purposes

    def __init__(self, data):
        self.data = data # data stored by the block

    def hash(self):
        h = hashlib.sha256()
        h.update(
        str(self.nonce).encode('utf-8') +
        str(self.data).encode('utf-8') +
        str(self.previous_hash).encode('utf-8') +
        str(self.timestamp).encode('utf-8') +
        str(self.blockNo).encode('utf-8')
        )
        return h.hexdigest()

    def __str__(self):
        return "Block Hash: " + str(self.hash()) + \
               "\nBlockNo: " + str(self.blockNo) + \
               "\nBlock Data: " + str(self.data) + \
               "\nHashes: " + str(self.nonce) + \
               "\n--------------"

class Blockchain:

    diff = 20 # difficulty level. The higher, the lower the target range and since for a block
    #to be added to the chain its hash must be below the target value
    maxNonce = 2**32 # max 32-bit number
    target = 2 ** (256-diff)

    block = Block("Genesis")
    dummy = head = block

    def add(self, block): # grow the blockchain by adding new block

        block.previous_hash = self.block.hash()#connect the blocks. n
        block.blockNo = self.block.blockNo + 1 # increament

        self.block.next = block # the NEXT pointer references the new block
        self.block = self.block.next# # make the newly added block to be the current block
# mining involves checking whether a block's hash should be included in the block-chain. This is done by checking if the hash val is less than a target value
    def mine(self, block):
        for n in range(self.maxNonce):
            if int(block.hash(), 16) <= self.target:
                self.add(block)
                print(block)
                break # stop mining
            else:
                block.nonce += 1 # if the hash doesnt fall within the expecte range, we change the block's hash by changing the nonce value to ensure the next hash is new and not equal to previous

blockchain = Blockchain()

for n in range(10): #generate the said number of blocks
    blockchain.mine(Block("Block " + str(n+1)))

while blockchain.head != None: # print the blocks
    print(blockchain.head)
    blockchain.head = blockchain.head.next

    #https://www.youtube.com/watch?v=gSQXq2_j-mw
    #https://www.youtube.com/watch?v=gwitf7ABtK8
    #https://www.youtube.com/watch?v=X8SPO875mQY
    #https://www.youtube.com/watch?v=z2qOxVVAD-o