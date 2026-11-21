// Minimal self-contained header-only HTTP/1.1 Server (WinSock2) for PROMETHEVS v41_serve2.cpp
#pragma once

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace httplib {

struct Request {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

struct DataSink {
    using WriteFn = std::function<bool(const char *, size_t)>;
    using DoneFn  = std::function<void()>;

    WriteFn write = [](const char *, size_t) { return false; };
    DoneFn  done  = []() {};
};

struct Response {
    int status = 200;
    std::string body;
    std::string content_type = "text/plain";
    bool chunked = false;
    std::function<bool(size_t, DataSink &)> chunked_provider;

    void set_content(const std::string & s, const char * ct) {
        body = s;
        content_type = ct ? ct : "application/octet-stream";
        chunked = false;
        chunked_provider = nullptr;
    }

    void set_chunked_content_provider(const char * ct, std::function<bool(size_t, DataSink &)> provider) {
        content_type = ct ? ct : "text/event-stream";
        chunked = true;
        chunked_provider = std::move(provider);
    }
};

struct TaskQueue {
    TaskQueue() = default;
    virtual ~TaskQueue() = default;
    virtual bool enqueue(std::function<void()> fn) = 0;
    virtual void shutdown() = 0;
    virtual void on_idle() {}
};

class ThreadPool : public TaskQueue {
public:
    explicit ThreadPool(size_t n) : shutdown_(false) {
        if (n == 0) n = 1;
        for (size_t i = 0; i < n; ++i) {
            threads_.emplace_back([this]() { worker(); });
        }
    }

    ~ThreadPool() override {
        shutdown();
    }

    bool enqueue(std::function<void()> fn) override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_) return false;
            jobs_.push_back(std::move(fn));
        }
        cv_.notify_one();
        return true;
    }

    void shutdown() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (shutdown_) return;
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto & t : threads_) {
            if (t.joinable()) t.join();
        }
        threads_.clear();
    }

private:
    void worker() {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]() { return shutdown_ || !jobs_.empty(); });
                if (shutdown_ && jobs_.empty()) return;
                fn = std::move(jobs_.front());
                jobs_.pop_front();
            }
            if (fn) fn();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> jobs_;
    std::vector<std::thread> threads_;
    bool shutdown_;
};

class Server {
public:
    using Handler = std::function<void(const Request &, Response &)>;

    std::function<TaskQueue *(void)> new_task_queue;

    Server() {
        WSADATA wsa{};
        wsa_ok_ = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }

    ~Server() {
        stop();
        if (wsa_ok_) {
            WSACleanup();
            wsa_ok_ = false;
        }
    }

    Server & Get(const std::string & pattern, Handler h) {
        get_handlers_.push_back({ pattern, std::move(h) });
        return *this;
    }

    Server & Post(const std::string & pattern, Handler h) {
        post_handlers_.push_back({ pattern, std::move(h) });
        return *this;
    }

    void stop() {
        running_.store(false);
        SOCKET s = listen_sock_.exchange(INVALID_SOCKET);
        if (s != INVALID_SOCKET) closesocket(s);
    }

    bool listen(const char * host, int port) {
        if (!wsa_ok_) return false;

        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == INVALID_SOCKET) return false;

        int opt = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (!host || !*host || std::strcmp(host, "0.0.0.0") == 0) {
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
        } else {
            if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
                closesocket(s);
                return false;
            }
        }

        if (::bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == SOCKET_ERROR ||
            ::listen(s, SOMAXCONN) == SOCKET_ERROR) {
            closesocket(s);
            return false;
        }

        listen_sock_.store(s);
        running_.store(true);

        std::unique_ptr<TaskQueue> tq(new_task_queue ? new_task_queue() : new ThreadPool(8));
        while (running_.load()) {
            SOCKET client = accept(s, nullptr, nullptr);
            if (client == INVALID_SOCKET) {
                break;
            }
            int nodelay = 1;
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&nodelay), sizeof(nodelay));
            if (!tq->enqueue([this, client]() { handle_client(client); })) {
                closesocket(client);
            }
        }

        stop();
        tq->shutdown();
        return true;
    }

private:
    struct Route {
        std::string pattern;
        Handler handler;
    };

    static std::string to_lower(std::string s) {
        for (char & c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static std::string trim(const std::string & s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
        return s.substr(a, b - a);
    }

    static const char * status_text(int status) {
        switch (status) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 409: return "Conflict";
            case 500: return "Internal Server Error";
            default:  return "OK";
        }
    }

    static bool send_all(SOCKET sock, const char * data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            int chunk = static_cast<int>(std::min<size_t>(len - sent, 1 << 20));
            int n = send(sock, data + sent, chunk, 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    static bool decode_chunked_body(const std::string & raw, size_t start, std::string & out) {
        size_t pos = start;
        out.clear();
        for (;;) {
            size_t line_end = raw.find("\r\n", pos);
            if (line_end == std::string::npos) return false;
            std::string sz_str = raw.substr(pos, line_end - pos);
            size_t semi = sz_str.find(';');
            if (semi != std::string::npos) sz_str.resize(semi);
            unsigned long chunk_sz = std::strtoul(sz_str.c_str(), nullptr, 16);
            pos = line_end + 2;
            if (chunk_sz == 0) {
                return raw.find("\r\n", pos) != std::string::npos || pos + 2 <= raw.size();
            }
            if (pos + chunk_sz + 2 > raw.size()) return false;
            out.append(raw, pos, chunk_sz);
            pos += chunk_sz + 2;
        }
    }

    static bool read_request(SOCKET sock, Request & req) {
        std::string buf;
        buf.reserve(4096);
        char tmp[4096];
        size_t hdr_end = std::string::npos;
        while (hdr_end == std::string::npos) {
            int n = recv(sock, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, static_cast<size_t>(n));
            hdr_end = buf.find("\r\n\r\n");
            if (buf.size() > 64u * 1024u * 1024u) return false;
        }

        size_t line_end = buf.find("\r\n");
        if (line_end == std::string::npos) return false;
        std::string req_line = buf.substr(0, line_end);
        size_t sp1 = req_line.find(' ');
        size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : req_line.find(' ', sp1 + 1);
        if (sp1 == std::string::npos) return false;
        req.method = req_line.substr(0, sp1);
        std::string target = (sp2 == std::string::npos)
            ? req_line.substr(sp1 + 1)
            : req_line.substr(sp1 + 1, sp2 - sp1 - 1);
        size_t qm = target.find('?');
        req.path = (qm == std::string::npos) ? target : target.substr(0, qm);

        size_t pos = line_end + 2;
        size_t content_length = 0;
        bool has_cl = false;
        bool is_chunked = false;
        while (pos < hdr_end) {
            size_t next = buf.find("\r\n", pos);
            if (next == std::string::npos || next > hdr_end) break;
            std::string hline = buf.substr(pos, next - pos);
            pos = next + 2;
            size_t colon = hline.find(':');
            if (colon == std::string::npos) continue;
            std::string k = to_lower(trim(hline.substr(0, colon)));
            std::string v = trim(hline.substr(colon + 1));
            req.headers[k] = v;
            if (k == "content-length") {
                content_length = static_cast<size_t>(std::strtoull(v.c_str(), nullptr, 10));
                has_cl = true;
            } else if (k == "transfer-encoding" && to_lower(v).find("chunked") != std::string::npos) {
                is_chunked = true;
            }
        }

        size_t body_start = hdr_end + 4;
        if (is_chunked) {
            while (!decode_chunked_body(buf, body_start, req.body)) {
                int n = recv(sock, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                buf.append(tmp, static_cast<size_t>(n));
                if (buf.size() > 64u * 1024u * 1024u) return false;
            }
        } else if (has_cl && content_length > 0) {
            size_t have = buf.size() - body_start;
            while (have < content_length) {
                int n = recv(sock, tmp, sizeof(tmp), 0);
                if (n <= 0) return false;
                buf.append(tmp, static_cast<size_t>(n));
                have = buf.size() - body_start;
            }
            req.body.assign(buf, body_start, content_length);
        } else {
            req.body.clear();
        }
        return true;
    }

    const Handler * find_handler(const std::string & method, const std::string & path) const {
        const auto & routes = (method == "GET") ? get_handlers_ : post_handlers_;
        for (const auto & r : routes) {
            if (r.pattern == path) return &r.handler;
        }
        return nullptr;
    }

    void handle_client(SOCKET sock) {
        Request req;
        if (read_request(sock, req)) {
            Response res;
            const Handler * h = (req.method == "GET" || req.method == "POST")
                ? find_handler(req.method, req.path)
                : nullptr;

            if (!h) {
                res.status = 404;
                res.set_content("{\"error\":\"not found\"}", "application/json");
            } else {
                try {
                    (*h)(req, res);
                } catch (const std::exception & ex) {
                    res.status = 500;
                    res.set_content(std::string("{\"error\":\"") + ex.what() + "\"}", "application/json");
                } catch (...) {
                    res.status = 500;
                    res.set_content("{\"error\":\"internal error\"}", "application/json");
                }
            }

            if (res.chunked && res.chunked_provider) {
                char hdr[512];
                int hlen = std::snprintf(
                    hdr, sizeof(hdr),
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                    res.status, status_text(res.status), res.content_type.c_str());
                if (hlen > 0 && send_all(sock, hdr, static_cast<size_t>(hlen))) {
                    bool chunk_done = false;
                    bool broken = false;
                    DataSink sink;
                    sink.write = [sock, &broken](const char * data, size_t len) -> bool {
                        if (broken) return false;
                        if (len == 0) return true;
                        char pre[32];
                        int plen = std::snprintf(pre, sizeof(pre), "%zx\r\n", len);
                        if (plen <= 0 ||
                            !send_all(sock, pre, static_cast<size_t>(plen)) ||
                            !send_all(sock, data, len) ||
                            !send_all(sock, "\r\n", 2)) {
                            broken = true;
                            return false;
                        }
                        return true;
                    };
                    sink.done = [sock, &chunk_done, &broken]() {
                        if (!chunk_done && !broken) {
                            chunk_done = true;
                            send_all(sock, "0\r\n\r\n", 5);
                        }
                    };
                    res.chunked_provider(0, sink);
                    sink.done();
                }
            } else {
                char hdr[512];
                int hlen = std::snprintf(
                    hdr, sizeof(hdr),
                    "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                    res.status, status_text(res.status), res.content_type.c_str(), res.body.size());
                if (hlen > 0 && send_all(sock, hdr, static_cast<size_t>(hlen)) && !res.body.empty()) {
                    send_all(sock, res.body.data(), res.body.size());
                }
            }
        }
        shutdown(sock, SD_SEND);
        closesocket(sock);
    }

    std::vector<Route> get_handlers_;
    std::vector<Route> post_handlers_;
    std::atomic<SOCKET> listen_sock_{ INVALID_SOCKET };
    std::atomic<bool> running_{ false };
    bool wsa_ok_ = false;
};

} // namespace httplib
