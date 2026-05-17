class Agenticrepl < Formula
  include Language::Python::Virtualenv

  desc "Smart terminal REPL that thinks with you — natural language to executable commands"
  homepage "https://github.com/panlanfeng/AgenticREPL"
  url "https://github.com/panlanfeng/AgenticREPL/archive/refs/tags/v0.1.1.tar.gz"
  version "0.1.1"
  sha256 "b1fc9558c6b84dc6cc272b6014445a23dee43585d6bb8601fb5aaa4b23cf6ace"
  license "MIT"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/srun", "--help"
  end
end
