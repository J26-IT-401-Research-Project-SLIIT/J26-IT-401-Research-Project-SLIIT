import argparse, json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description="Component 4 PP1 fusion prototype")
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("train",help="Generate synthetic data, train and evaluate")
    commands.add_parser("audit",help="Audit the active pipeline and simulator sensitivity")
    serve=commands.add_parser("serve",help="Run the local demonstration/API"); serve.add_argument("--port",type=int,default=8040)
    predict=commands.add_parser("predict")
    for name in ["motor","voice","handwriting"]: predict.add_argument("--"+name,type=float)
    predict.add_argument("--lime",action="store_true")
    batch=commands.add_parser("batch"); batch.add_argument("--input",type=Path,required=True); batch.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="train":
        from .train import train
        train()
        from .validation import generate_explanation_evidence
        generate_explanation_evidence()
    elif args.command=="serve":
        import uvicorn
        uvicorn.run("multimodal_fusion.api:app",host="127.0.0.1",port=args.port)
    elif args.command=="audit":
        from .audit import audit
        audit()
    else:
        from .engine import FusionEngine
        from .settings import FEATURES,LABELS
        engine=FusionEngine()
        if args.command=="predict":
            print(json.dumps(engine.predict(dict(zip(FEATURES,[args.motor,args.voice,args.handwriting])),include_lime=args.lime),indent=2))
        else:
            import pandas as pd
            frame=pd.read_csv(args.input); missing=set(FEATURES)-set(frame.columns)
            if missing: raise ValueError(f"Missing CSV columns: {sorted(missing)}")
            rows=[]
            for i,row in frame.iterrows():
                scores={key:None if pd.isna(row[key]) else float(row[key]) for key in FEATURES}
                result=engine.predict(scores,session_id=f"batch-{i+1}")
                rows.append({"row":i+1,"status":result["status"],"predicted_class":result["predicted_class"],
                    **{f"p_{label}":(result["likeness_profile"] or {}).get(label) for label in LABELS}})
            pd.DataFrame(rows).to_csv(args.output,index=False); print(f"Wrote {len(rows)} predictions to {args.output}")

if __name__=="__main__": main()
