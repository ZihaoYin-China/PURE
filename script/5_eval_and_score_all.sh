
#!/bin/bash

# 自动定位到根目录
if [[ "$PWD" == */script ]]; then cd ..; fi

# 定义所有支持的数据集
TARGETS=("mmlu" "squad" "natural_questions" "hotpotqa" "webqa")
ROUTERS=("t5-large" "distilbert" "qwen")
MODEL="qwen2.5vl:32b"

echo "========================================================"
echo "开启 PURE 对比流水线"
echo "========================================================"

# ----------------- 阶段 1：生成结果 -----------------
for router in "${ROUTERS[@]}"; do
    echo -e "\n\nRunning router: [ $router ]"
    if [ ! -d "route/results/$router" ]; then
        echo "Skip: missing route/results/$router"
        continue
    fi

    for target in "${TARGETS[@]}"; do
        echo -e "  Evaluating dataset: [ $target ]"
        bash script/4_eval.sh --router_model "$router" --target "$target"
    done
done

echo -e "\n========================================================"
echo "All generations finished. Start scoring..."
echo "========================================================"

# ----------------- 阶段 2：打分 -----------------
for router in "${ROUTERS[@]}"; do
    echo -e "\n\nRouter [ $router ] summary:"
    echo "--------------------------------------------------------"
    for target in "${TARGETS[@]}"; do
        FILE_PATH="eval/results/${MODEL}/${router}/${target}_top1_0.2_1.json"
        
        if [ -f "$FILE_PATH" ]; then
            echo "📊 数据集: $target"
            # 只提取核心分数，让控制台看起来像个真正的计分板
            python eval/score.py --result_file "$FILE_PATH" --target "$target" | grep -E "Accuracy|EM|F1|ROUGE-L|BERTScore|Count"
            echo ""
        fi
    done
    echo "========================================================"
done

echo "Done."
