import clip
from PIL import Image
import base64
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct

from anubis_core.ports.db import IVectorSearchPort


class QdrantVectorAdapter(IVectorSearchPort):
    def __init__(self, host: str, port: int, collection_name: str):
        import clip
        import torch
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # if not self.client.collection_exists(collection_name):
        #     self.client.create_collection(
        #         collection_name=collection_name,
        #         vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
        #     )
        #     print(f"✅ Colección '{collection_name}' creada.")


    def create_embedding(self, text: str) -> list[float]:        
        import torch
        image_data = base64.b64decode(text)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features[0].cpu().numpy().tolist()

    def search_similar(self, vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k
        )
        return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]

    def index_document(self, id: str, vector: list[float], metadata: dict) -> bool:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": id,
                "vector": vector,
                "payload": metadata
            }]
        )
        return True
    