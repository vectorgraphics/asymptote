# LanguageSession 关闭契约

本文档描述 LspCpp 当前的 shutdown 行为，用于记录现有语义，避免 embedder 误用
clangd 特有的 exit code 规则。

## 期望客户端顺序

1. `initialize` 请求/响应
2. 可选 `initialized` 通知
3. 正常 LSP 流量
4. `shutdown` 请求，返回 `result: null`
5. `exit` 通知

## 当前 LspCpp 行为

| 场景 | 当前行为 |
| --- | --- |
| 正常 `shutdown` 后 `exit` | `exit` 通知 handler 会执行，会话正常结束。 |
| 未 `shutdown` 直接 `exit` | `exit` 通知 handler 仍会执行；LspCpp 不强制进程退出码。 |
| 首帧前 stdin EOF | 消息读取线程结束；不会派发 `exit` 通知。 |
| 带 id 的不支持请求 | 未注册 handler/parser 时返回 `-32601`。 |
| endpoint `stop()` | 未完成 outbound 请求失败，处理线程 join。 |

## 可选限制

`RemoteEndPointLimits::max_pending_outgoing_requests` 默认为 `0`（无限制）。
设置后，pending reply 映射达到上限时会拒绝新的 server→client 请求。

## 兼容性说明

这些语义由借鉴 clangd 场景的 characterization tests 覆盖。
若未来调整行为，需同步更新本文档与对应测试。
