class Agenticrepl < Formula
  include Language::Python::Virtualenv

  desc "Smart terminal REPL that thinks with you — natural language to executable commands"
  homepage "https://github.com/panlanfeng/AgenticREPL"
  url "https://github.com/panlanfeng/AgenticREPL/archive/refs/heads/main.tar.gz"
  version "0.1.0"
  sha256 "d32bf8bbc1430f8c9e813c0ebd249f4dc2838a181fe3220c2d09e2f026a31001"
  license "MIT"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/srun", "--help"
  end
end
